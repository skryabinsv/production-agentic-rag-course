"""API-key authentication and rate limiting (src/security.py)."""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from src import dependencies
from src.config import Settings
from src.security import _MemoryWindowCounter


def _app(settings: Settings) -> FastAPI:
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(dependencies.protect_api)])
    def protected():
        return {"ok": True}

    app.dependency_overrides[dependencies.get_settings] = lambda: settings
    return app


def test_no_api_keys_means_open_access():
    client = TestClient(_app(Settings(api_keys="", rate_limit_per_minute=0)))
    assert client.get("/protected").status_code == 200


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({}, 401),
        ({"X-API-Key": "wrong"}, 401),
        ({"X-API-Key": "secret-1"}, 200),
        ({"Authorization": "Bearer secret-2"}, 200),
    ],
)
def test_api_key_required_when_configured(headers, expected):
    client = TestClient(_app(Settings(api_keys="secret-1, secret-2", rate_limit_per_minute=0)))
    assert client.get("/protected", headers=headers).status_code == expected


def test_rate_limit_returns_429_with_retry_after():
    client = TestClient(_app(Settings(api_keys="k", rate_limit_per_minute=2)))
    headers = {"X-API-Key": "k"}
    assert client.get("/protected", headers=headers).status_code == 200
    assert client.get("/protected", headers=headers).status_code == 200
    response = client.get("/protected", headers=headers)
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1


def test_memory_counter_resets_each_window():
    counter = _MemoryWindowCounter()
    assert [counter.hit("a", 1), counter.hit("a", 1), counter.hit("a", 2)] == [1, 2, 1]
