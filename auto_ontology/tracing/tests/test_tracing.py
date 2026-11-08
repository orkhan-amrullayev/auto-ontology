# SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trace context and its transport to the proxy."""

from __future__ import annotations

import threading

import httpx
import pytest

from auto_ontology.tracing import context as tc
from auto_ontology.tracing import transport


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("LLM_TRACE_PROXY_URL", "http://proxy:3004/")
    return "http://proxy:3004"


def test_nothing_is_current_outside_a_trace() -> None:
    assert tc.current() is None
    with tc.span("step"), tc.call_metadata(schema="X"):
        assert tc.current() is None


def test_spans_nest_and_unwind() -> None:
    with tc.start_trace("chat", user_id="u1") as trace:
        with tc.span("a"):
            with tc.span("b"):
                ctx = tc.current()
                assert ctx is not None
                assert [s.name for s in ctx.spans] == ["a", "b"]
                assert ctx.trace_id == trace.trace_id
                assert ctx.user_id == "u1"
            inner = tc.current()
            assert inner is not None and [s.name for s in inner.spans] == ["a"]
    assert tc.current() is None


def test_call_metadata_merges_for_the_block_only() -> None:
    with tc.start_trace("chat"), tc.call_metadata(schema="S"):
        with tc.call_metadata(attempt=2):
            ctx = tc.current()
            assert ctx is not None and ctx.call == {"schema": "S", "attempt": 2}
        ctx = tc.current()
        assert ctx is not None and ctx.call == {"schema": "S"}


def test_ensure_trace_joins_an_existing_trace() -> None:
    with tc.start_trace("chat") as outer:
        with tc.ensure_trace("api") as joined:
            assert joined.trace_id == outer.trace_id
    with tc.ensure_trace("api") as fresh:
        assert fresh.kind == "api"


def test_payload_round_trips_across_a_process_hop() -> None:
    with tc.start_trace("chat", conversation_id="c1", title="q"), tc.span("n"):
        ctx = tc.current()
        assert ctx is not None
        restored = tc.TraceContext.from_payload(ctx.to_payload())
    assert restored == ctx


def test_restore_never_raises_on_a_bad_payload() -> None:
    assert tc.restore(None) is None
    assert tc.restore({"unexpected": 1}) is None


def test_tracing_executor_carries_the_trace_but_plain_threads_do_not() -> None:
    seen: dict[str, object] = {}

    def probe(label: str) -> None:
        seen[label] = tc.current()

    with tc.start_trace("chat") as trace, tc.span("pool"):
        with tc.TracingThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(probe, "traced").result()
        plain = threading.Thread(target=probe, args=("plain",))
        plain.start()
        plain.join()

    traced = seen["traced"]
    assert isinstance(traced, tc.TraceContext) and traced.trace_id == trace.trace_id
    assert seen["plain"] is None


def test_route_is_a_no_op_when_tracing_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_TRACE_PROXY_URL", raising=False)
    url = "https://res.openai.azure.com/openai/v1/chat/completions"
    assert transport.route(url) == (url, {})
    assert transport.sync_http_client() is None
    assert transport.async_http_client() is None


def test_route_targets_the_proxy_and_names_the_upstream(proxy: str) -> None:
    url = "https://res.openai.azure.com/openai/v1/embeddings?api-version=1"
    with tc.start_trace("ingest", database_name="wwi"):
        target, headers = transport.route(url)
    assert target == f"{proxy}/forward/openai/v1/embeddings?api-version=1"
    assert headers[transport.UPSTREAM_HEADER] == "https://res.openai.azure.com"
    decoded = transport.decode_trace(headers[transport.TRACE_HEADER])
    assert decoded["kind"] == "ingest" and decoded["database_name"] == "wwi"


def test_route_without_a_trace_still_reaches_the_proxy(proxy: str) -> None:
    target, headers = transport.route("https://api.example/v1/chat/completions")
    assert target.startswith(proxy)
    assert transport.TRACE_HEADER not in headers


def test_the_trace_header_survives_non_ascii_text(proxy: str) -> None:
    with tc.start_trace("chat", title="Hvad var omsætningen?\nlinje 2"):
        _, headers = transport.route("https://api.example/v1/chat/completions")
    value = headers[transport.TRACE_HEADER]
    assert value.isascii() and "\n" not in value
    assert transport.decode_trace(value)["title"] == "Hvad var omsætningen?\nlinje 2"


def test_the_httpx_hook_reroutes_a_real_request(proxy: str) -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        event_hooks={"request": [transport._reroute]},
    )
    with tc.start_trace("chat"), tc.call_metadata(schema="SQLGenerationModel"):
        client.post("https://res.openai.azure.com/openai/v1/chat/completions", json={})

    [request] = sent
    assert str(request.url) == f"{proxy}/forward/openai/v1/chat/completions"
    assert request.headers["host"] == "proxy:3004"
    trace = transport.decode_trace(request.headers[transport.TRACE_HEADER])
    assert trace["call"] == {"schema": "SQLGenerationModel"}
