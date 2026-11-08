# SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trace proxy: capture helpers and the forwarding route."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from auto_ontology.llm_trace_proxy import app as proxy_app
from auto_ontology.llm_trace_proxy import capture
from auto_ontology.tracing import context as tc
from auto_ontology.tracing import transport

UPSTREAM = "https://llm.example.com"


def test_endpoints_are_classified_from_the_path() -> None:
    assert capture.classify("/openai/v1/chat/completions") == "chat"
    assert capture.classify("openai/v1/embeddings") == "embedding"
    assert capture.classify("/v1/ranking") == "rerank"
    assert capture.classify("/openai/deployments") == "other"


def test_embedding_vectors_are_not_stored() -> None:
    body = {"data": [{"index": 0, "embedding": [0.1] * 1536}], "usage": {}}
    stored = capture.redact_response("embedding", body)
    assert stored["data"] == [{"index": 0, "embedding_dims": 1536}]


def test_long_strings_are_truncated() -> None:
    stored = capture.redact_request({"input": ["x" * (capture.MAX_STRING_CHARS + 5)]})
    assert stored["input"][0].endswith("[truncated 5 chars]")


def test_usage_is_read_and_totalled() -> None:
    assert capture.usage({"usage": {"prompt_tokens": 3, "completion_tokens": 4}}) == (
        3,
        4,
        7,
    )
    assert capture.usage({"no": "usage"}) == (None, None, None)


def test_a_stream_is_reassembled() -> None:
    chunks = [
        {"model": "gpt-x", "choices": [{"index": 0, "delta": {"content": "Hel"}}]},
        {
            "choices": [
                {"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}},
    ]
    raw = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    result = capture.assemble_stream(raw.encode())
    assert result["content"] == "Hello"
    assert result["model"] == "gpt-x"
    assert result["finish_reason"] == "stop"
    assert capture.usage(result) == (1, 2, 3)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The proxy app with a fake upstream and an in-memory store."""
    monkeypatch.setenv("DEFAULT_MODELS_ENDPOINT", f"{UPSTREAM}/openai/v1/")
    recorded: list[dict[str, Any]] = []
    forwarded: list[httpx.Request] = []
    # Rows are normally written on an executor; capture them inline instead.
    monkeypatch.setattr(proxy_app, "_dispatch", recorded.append)

    def upstream(request: httpx.Request) -> httpx.Response:
        forwarded.append(request)
        return httpx.Response(
            200,
            json={
                "model": "gpt-5-mini",
                "choices": [{"message": {"content": '{"sql": "SELECT 1"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    real_client = httpx.AsyncClient

    def fake_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(upstream), **kwargs)

    monkeypatch.setattr(proxy_app.httpx, "AsyncClient", fake_client)
    return {"recorded": recorded, "forwarded": forwarded}


def test_a_call_is_forwarded_and_recorded_with_its_trace(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_TRACE_PROXY_URL", "http://proxy")
    with tc.start_trace("chat", user_id="u1", title="revenue?"), tc.span("sql"):
        with tc.call_metadata(schema="SQLGenerationModel"):
            trace_header = transport.encode_trace()
    assert trace_header

    with TestClient(proxy_app.create_app()) as client:
        response = client.post(
            "/forward/openai/v1/chat/completions",
            headers={
                transport.UPSTREAM_HEADER: UPSTREAM,
                transport.TRACE_HEADER: trace_header,
                "api-key": "secret",
            },
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "q"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["usage"]["prompt_tokens"] == 10
    [sent] = harness["forwarded"]
    assert str(sent.url) == f"{UPSTREAM}/openai/v1/chat/completions"
    assert sent.headers["api-key"] == "secret"  # forwarded, never stored
    assert transport.TRACE_HEADER.lower() not in sent.headers

    [row] = harness["recorded"]
    assert row["endpoint"] == "chat" and row["model"] == "gpt-5-mini"
    assert row["user_id"] == "u1" and row["title"] == "revenue?"
    assert [s["name"] for s in row["spans"]] == ["sql"]
    assert row["call"] == {"schema": "SQLGenerationModel"}
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (
        10,
        5,
        15,
    )
    assert "secret" not in json.dumps(row, default=str)


def test_an_unlisted_upstream_is_refused(harness: dict[str, Any]) -> None:
    with TestClient(proxy_app.create_app()) as client:
        response = client.post(
            "/forward/v1/chat/completions",
            headers={transport.UPSTREAM_HEADER: "https://evil.example"},
            json={},
        )
    assert response.status_code == 403
    assert harness["forwarded"] == [] and harness["recorded"] == []
