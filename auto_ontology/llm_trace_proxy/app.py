# SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
# SPDX-License-Identifier: Apache-2.0

"""The trace proxy's HTTP app.

``/forward/{path}`` sends the request on to ``X-AO-Upstream`` + ``/{path}``
with the caller's own headers (minus hop-by-hop ones and the ``X-AO-*`` ones),
streams the answer back as it arrives, and records the exchange afterwards.

Only *allowed* upstreams are forwarded to, so the proxy cannot be used to reach
arbitrary hosts: the origins of the configured model endpoints
(``REASONING``/``NON_REASONING``/``EMBED``/``RERANK``, resolved exactly as the
app resolves them) plus any listed in ``LLM_TRACE_PROXY_ALLOWED_UPSTREAMS``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from auto_ontology.llm_trace_proxy import capture
from auto_ontology.llm_trace_proxy.store import record_call
from auto_ontology.tracing.transport import TRACE_HEADER, UPSTREAM_HEADER, decode_trace
from auto_ontology.utils.model_config import resolve

logger = logging.getLogger(__name__)

# Never forwarded upstream or back: connection-level, recomputed by httpx, or ours.
_DROP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "te",
    "upgrade",
    "proxy-connection",
    UPSTREAM_HEADER.lower(),
    TRACE_HEADER.lower(),
}
# httpx hands back decoded bytes, so the upstream's encoding/length no longer apply.
_DROP_RESPONSE_HEADERS = {
    "content-encoding",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
}
_TIMEOUT = httpx.Timeout(float(os.environ.get("LLM_TRACE_PROXY_TIMEOUT_S", "900")))


def _origin(url: str) -> str | None:
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}".lower()


def allowed_upstreams() -> set[str]:
    origins = {
        _origin(resolve(prefix, "ENDPOINT"))
        for prefix in ("REASONING", "NON_REASONING", "EMBED", "RERANK")
    }
    extra = os.environ.get("LLM_TRACE_PROXY_ALLOWED_UPSTREAMS", "")
    origins.update(_origin(item) for item in extra.split(",") if item.strip())
    return {origin for origin in origins if origin}


def _trace_fields(header: str | None) -> dict[str, Any]:
    """Columns from the ``X-AO-Trace`` header; empty when absent or malformed."""
    if not header:
        return {}
    try:
        trace = decode_trace(header)
    except Exception:  # noqa: BLE001 — a bad header still gets the call recorded
        logger.warning("Ignoring malformed %s header", TRACE_HEADER)
        return {}
    return {
        "trace_id": trace.get("trace_id"),
        "kind": trace.get("kind"),
        "user_id": trace.get("user_id"),
        "conversation_id": trace.get("conversation_id"),
        "turn_id": trace.get("turn_id"),
        "source": trace.get("source"),
        "database_name": trace.get("database_name"),
        "title": trace.get("title"),
        "spans": trace.get("spans") or [],
        "call": trace.get("call") or {},
    }


def _row(
    *,
    request: Request,
    upstream: str,
    path: str,
    trace: dict[str, Any],
    request_body: Any,
    started_at: datetime,
    started: float,
    status_code: int | None,
    response_body: Any,
    streamed: bool,
    error: str | None,
) -> dict[str, Any]:
    endpoint = capture.classify(path)
    prompt, completion, total = capture.usage(response_body)
    model = request_body.get("model") if isinstance(request_body, dict) else None
    if not model and isinstance(response_body, dict):
        model = response_body.get("model")
    return {
        **trace,
        "started_at": started_at,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "endpoint": endpoint,
        "method": request.method,
        "upstream": upstream,
        "path": f"/{path}",
        "model": model,
        "streamed": streamed,
        "status_code": status_code,
        "error": error,
        "request": capture.redact_request(request_body),
        "response": capture.redact_response(endpoint, response_body),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _dispatch(row: dict[str, Any]) -> None:
    """Write *row* off the event loop, so a slow database never delays a reply."""
    asyncio.get_running_loop().run_in_executor(None, record_call, row)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.client = httpx.AsyncClient(timeout=_TIMEOUT)
        app.state.allowed = allowed_upstreams()
        logger.info(
            "LLM trace proxy forwarding to: %s", ", ".join(sorted(app.state.allowed))
        )
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="Auto Ontology LLM trace proxy", lifespan=lifespan)

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route(
        "/forward/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"]
    )
    async def forward(path: str, request: Request) -> Response:
        upstream = _origin(request.headers.get(UPSTREAM_HEADER, "") or "")
        if upstream is None or upstream not in request.app.state.allowed:
            return JSONResponse(
                {"error": f"upstream not allowed: {upstream or '(missing)'}"},
                status_code=403,
            )
        trace = _trace_fields(request.headers.get(TRACE_HEADER))
        raw_body = await request.body()
        request_body = capture.parse_json(raw_body)
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _DROP_REQUEST_HEADERS
        }
        target = f"{upstream}/{path}"
        if request.url.query:
            target += f"?{request.url.query}"

        client: httpx.AsyncClient = request.app.state.client
        started_at = datetime.now(timezone.utc)
        started = time.monotonic()

        def record(**outcome: Any) -> None:
            row = _row(
                request=request,
                upstream=upstream,
                path=path,
                trace=trace,
                request_body=request_body,
                started_at=started_at,
                started=started,
                **outcome,
            )
            _dispatch(row)

        try:
            upstream_response = await client.send(
                client.build_request(
                    request.method, target, headers=headers, content=raw_body
                ),
                stream=True,
            )
        except httpx.HTTPError as exc:
            record(
                status_code=None,
                response_body=None,
                streamed=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            return JSONResponse(
                {"error": f"upstream unreachable: {type(exc).__name__}"},
                status_code=502,
            )

        response_headers = {
            key: value
            for key, value in upstream_response.headers.items()
            if key.lower() not in _DROP_RESPONSE_HEADERS
        }
        is_stream = "text/event-stream" in upstream_response.headers.get(
            "content-type", ""
        )

        if not is_stream:
            content = await upstream_response.aread()
            await upstream_response.aclose()
            body = capture.parse_json(content)
            record(
                status_code=upstream_response.status_code,
                response_body=body,
                streamed=False,
                error=None if upstream_response.is_success else _error_text(body),
            )
            return Response(
                content=content,
                status_code=upstream_response.status_code,
                headers=response_headers,
            )

        async def relay() -> AsyncIterator[bytes]:
            collected = bytearray()
            error: str | None = None
            try:
                async for chunk in upstream_response.aiter_bytes():
                    collected.extend(chunk)
                    yield chunk
            except Exception as exc:  # noqa: BLE001 — recorded, then re-raised
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                await upstream_response.aclose()
                record(
                    status_code=upstream_response.status_code,
                    response_body=capture.assemble_stream(bytes(collected)),
                    streamed=True,
                    error=error,
                )

        return StreamingResponse(
            relay(),
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    return app


def _error_text(body: Any) -> str:
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        error = body["error"]
        return str(error.get("message") or error.get("code") or error)[:2000]
    return str(body)[:2000]
