# SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LLM Tracing API routes (DAL stubbed)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.server.traces import router as traces_router

NOW = datetime(2026, 11, 8, 21, 0, tzinfo=timezone.utc)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    seen: dict[str, Any] = {}

    def list_traces(**kwargs: Any) -> list[dict[str, Any]]:
        seen.update(kwargs)
        return [
            {
                "trace_id": "t1",
                "kind": "chat",
                "title": "revenue?",
                "user_email": "a@b.c",
                "started_at": NOW,
                "calls": 3,
                "llm_calls": 2,
                "embedding_calls": 1,
                "errors": 0,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "models": ["gpt-5-mini"],
            }
        ]

    def get_trace_calls(trace_id: str) -> list[dict[str, Any]]:
        if trace_id != "t1":
            return []
        return [
            {
                "id": 1,
                "started_at": NOW,
                "trace_id": "t1",
                "spans": [{"id": "s", "name": "construct_sql_from_candidates"}],
                "call": {"schema": "SQLGenerationModel"},
                "endpoint": "chat",
                "method": "POST",
                "upstream": "https://res.openai.azure.com",
                "path": "/openai/v1/chat/completions",
                "request": {"messages": []},
                "response": {"choices": []},
            }
        ]

    monkeypatch.setattr(traces_router.traces_dal, "list_traces", list_traces)
    monkeypatch.setattr(traces_router.traces_dal, "get_trace_calls", get_trace_calls)
    app = FastAPI()
    app.include_router(traces_router.router, prefix="/api")
    test_client = TestClient(app)
    test_client.seen = seen  # type: ignore[attr-defined]
    return test_client


def test_traces_are_listed_with_filters_passed_through(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_TRACE_PROXY_URL", "http://proxy:3004")
    response = client.get("/api/traces", params={"kind": "chat", "search": "rev"})

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["items"][0]["trace_id"] == "t1"
    assert client.seen["kind"] == "chat"  # type: ignore[attr-defined]
    assert client.seen["search"] == "rev"  # type: ignore[attr-defined]


def test_a_trace_returns_its_calls(client: TestClient) -> None:
    response = client.get("/api/traces/t1")

    assert response.status_code == 200
    [call] = response.json()["calls"]
    assert call["call"]["schema"] == "SQLGenerationModel"
    assert call["spans"][0]["name"] == "construct_sql_from_candidates"


def test_an_unknown_trace_is_404(client: TestClient) -> None:
    assert client.get("/api/traces/nope").status_code == 404
