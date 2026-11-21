import fcntl
import logging
import os
import tempfile
from typing import Iterable, Optional

from src.schemas.api.ask import AskRequest, AskResponse
from src.services.retrieval import Retriever
from src.utils.arxiv import arxiv_abs_url, arxiv_pdf_url
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logger = logging.getLogger(__name__)

# Telegram allows a single getUpdates consumer per bot token. The API image runs uvicorn with
# several workers and each worker executes the FastAPI lifespan, so without a lock every worker
# started its own poller and they kept kicking each other off ("Conflict: terminated by other
# getUpdates request"). A process-lifetime file lock elects exactly one poller per container.
_LOCK_PATH = os.path.join(tempfile.gettempdir(), "arxiv-curator-telegram-bot.lock")

# Telegram rejects messages longer than 4096 characters.
_TELEGRAM_MAX_MESSAGE_LEN = 4096


class TelegramBot:
    """Simple Telegram bot for Q&A."""

    def __init__(
        self,
        bot_token: str,
        opensearch_client,
        embeddings_client,
        ollama_client,
        cache_client=None,
        model: str = "llama3.2:1b",
        allowed_user_ids: Optional[Iterable[int]] = None,
        allow_public: bool = False,
        retriever: Optional[Retriever] = None,
    ):
        """Initialize bot with required services.

        :param model: Ollama model used for answers (was hard-coded)
        :param allowed_user_ids: Telegram user ids allowed to use the bot
        :param allow_public: Answer everyone when the allow-list is empty (off by default)
        :param retriever: Shared retrieval path (threaded OpenSearch, BM25 fallback, reranking)
        """
        self.bot_token = bot_token
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.ollama = ollama_client
        self.cache = cache_client
        self.model = model
        self.allowed_user_ids = set(allowed_user_ids or [])
        self.allow_public = allow_public
        self.retriever = retriever or Retriever(opensearch_client, embeddings_client)
        self.application: Optional[Application] = None
        self._lock_file = None

    def _acquire_single_instance_lock(self) -> bool:
        """Try to become the only polling process in this container."""
        try:
            lock_file = open(_LOCK_PATH, "w")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        self._lock_file = lock_file
        return True

    def _release_single_instance_lock(self) -> None:
        if self._lock_file:
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
            finally:
                self._lock_file = None

    def _is_allowed(self, update: Update) -> bool:
        if not self.allowed_user_ids:
            # Private by default: an empty allow-list no longer means "answer anyone"
            return self.allow_public
        user = update.effective_user
        return bool(user and user.id in self.allowed_user_ids)

    async def _reject(self, update: Update) -> None:
        user = update.effective_user
        user_id = user.id if user else "unknown"
        await update.message.reply_text(
            "Sorry, this bot is private.\n"
            f"Your Telegram user id is {user_id}. The bot owner can allow it via TELEGRAM__ALLOWED_USER_IDS."
        )

    async def start(self) -> bool:
        """Start the bot. Returns False if another worker already runs the poller."""
        if not self._acquire_single_instance_lock():
            logger.info("Telegram bot is already polling in another worker - skipping in this one")
            return False

        if not self.allowed_user_ids and self.allow_public:
            logger.warning("Telegram bot is PUBLIC (TELEGRAM__ALLOW_PUBLIC=true): it will answer anyone who finds it")
        elif not self.allowed_user_ids:
            logger.warning("TELEGRAM__ALLOWED_USER_IDS is empty: the bot will refuse everyone and tell them their user id")

        logger.info("Starting Telegram bot...")
        self.application = Application.builder().token(self.bot_token).build()

        # Register handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("search", self._search_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_question))

        # Start polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        logger.info("Telegram bot started successfully")
        return True

    async def stop(self) -> None:
        """Stop the bot."""
        try:
            if self.application:
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                logger.info("Telegram bot stopped")
        finally:
            self._release_single_instance_lock()

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        await update.message.reply_text(
            "Welcome to arXiv Paper Curator!\n\n"
            "Ask me questions about CS papers and I'll provide answers with sources.\n\n"
            "Commands:\n"
            "/help - Show this help\n"
            "/search <keywords> - Search papers"
        )

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        await update.message.reply_text(
            "Send me any question about computer science research papers.\n\n"
            "Examples:\n"
            "- What are transformer architectures?\n"
            "- How does BERT work?\n"
            "- Explain attention mechanisms\n\n"
            "Use /search to find specific papers."
        )

    async def _search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /search command."""
        if not self._is_allowed(update):
            await self._reject(update)
            return
        if not context.args:
            await update.message.reply_text("Usage: /search <keywords>\nExample: /search neural networks")
            return

        query = " ".join(context.args)
        await update.message.chat.send_action("typing")

        try:
            # Shared retrieval path (no reranking for a plain paper list)
            result = await self.retriever.retrieve(query, top_k=10, use_hybrid=True, rerank=False)
            hits = result.hits
            if not hits:
                await update.message.reply_text("No papers found. Try different keywords.")
                return

            # Deduplicate by arxiv_id (since chunks may have same paper)
            seen_ids = set()
            unique_papers = []
            for hit in hits:
                arxiv_id = hit.get("arxiv_id", "")
                if arxiv_id and arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    unique_papers.append(hit)
                if len(unique_papers) >= 5:
                    break

            # Format results
            response = f"Found {len(unique_papers)} papers:\n\n"
            for idx, hit in enumerate(unique_papers, 1):
                title = hit.get("title", "Untitled")
                arxiv_id = hit.get("arxiv_id", "")
                url = arxiv_abs_url(arxiv_id)
                response += f"{idx}. {title}\n{url}\n\n"

            await update.message.reply_text(response, disable_web_page_preview=True)

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            await update.message.reply_text("Search failed due to an internal error. Please try again later.")

    async def _handle_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle user questions."""
        if not self._is_allowed(update):
            await self._reject(update)
            return
        query = update.message.text
        await update.message.chat.send_action("typing")

        try:
            # Build request
            ask_request = AskRequest(query=query, top_k=3, use_hybrid=True, model=self.model)

            # Check cache
            if self.cache:
                try:
                    cached_response = await self.cache.find_cached_response(ask_request)
                    if cached_response:
                        await self._send_answer(update, cached_response)
                        return
                except Exception as e:
                    logger.warning(f"Cache lookup failed: {e}")

            # RAG pipeline
            from src.services.ollama.prompts import RAGPromptBuilder

            # Shared retrieval path: BM25 fallback, threaded OpenSearch, optional reranking
            result = await self.retriever.retrieve(query, top_k=ask_request.top_k, use_hybrid=ask_request.use_hybrid)

            # Extract chunks and sources
            chunks = []
            sources = []
            for hit in result.hits:
                arxiv_id = hit.get("arxiv_id", "")
                chunks.append(
                    {
                        "arxiv_id": arxiv_id,
                        "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                    }
                )
                if arxiv_id and arxiv_pdf_url(arxiv_id) not in sources:
                    sources.append(arxiv_pdf_url(arxiv_id))

            if not chunks:
                await update.message.reply_text("No relevant papers found. Try rephrasing your question.")
                return

            # Generate answer
            prompt = RAGPromptBuilder().create_rag_prompt(query=query, chunks=chunks)
            ollama_response = await self.ollama.generate(
                model=self.model, prompt=prompt, stream=False, temperature=getattr(self.ollama, "temperature", 0.2)
            )
            answer = ollama_response.get("response", "") if ollama_response else ""

            # Build response
            response = AskResponse(
                query=query,
                answer=answer,
                sources=sources,
                chunks_used=len(chunks),
                search_mode=result.search_mode,
            )

            # Cache it
            if self.cache:
                try:
                    await self.cache.store_response(ask_request, response)
                except Exception:
                    pass

            # Send to user
            await self._send_answer(update, response)

        except Exception as e:
            logger.error(f"Question handling failed: {e}", exc_info=True)
            await update.message.reply_text("Sorry, something went wrong while answering. Please try again later.")

    async def _send_answer(self, update: Update, response: AskResponse) -> None:
        """Send formatted answer to user."""
        # Answer
        message = f"*Answer:*\n{response.answer}\n"

        # Sources
        if response.sources:
            message += "\n*Sources:*\n"
            for idx, source_url in enumerate(response.sources[:5], 1):
                arxiv_id = source_url.split("/")[-1].replace(".pdf", "")
                message += f"{idx}. https://arxiv.org/abs/{arxiv_id}\n"

        if len(message) > _TELEGRAM_MAX_MESSAGE_LEN:
            message = message[: _TELEGRAM_MAX_MESSAGE_LEN - 20] + "\n\n[truncated]"

        # Send (try markdown, fallback to plain)
        try:
            await update.message.reply_text(message, parse_mode="Markdown", disable_web_page_preview=True)
        except Exception:
            await update.message.reply_text(message, disable_web_page_preview=True)
