# SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
# SPDX-License-Identifier: Apache-2.0

"""Route outbound model requests through the trace proxy.

With ``LLM_TRACE_PROXY_URL`` set, a request for
``https://res.openai.azure.com/openai/v1/chat/completions`` is sent to
``<proxy>/forward/openai/v1/chat/completions`` instead, carrying:

* ``X-AO-Upstream`` -- the original origin (``https://res.openai.azure.com``),
  which the proxy forwards to after checking it against its allowlist;
* ``X-AO-Trace`` -- the current :class:`~.context.TraceContext` as base64url
  JSON (one opaque header, so non-ASCII questions and newlines cannot break
  HTTP), absent outside a trace.

Without the variable nothing here changes a request, and the client factories
return ``None`` so callers build exactly the clients they did before.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any
from urllib.parse import urlsplit

import httpx

from auto_ontology.tracing.context import current

UPSTREAM_HEADER = "X-AO-Upstream"
TRACE_HEADER = "X-AO-Trace"
_FORWARD_PREFIX = "/forward"


def proxy_url() -> str:
    """The trace proxy's base URL, or ``""`` when tracing is off."""
    return os.environ.get("LLM_TRACE_PROXY_URL", "").strip().rstrip("/")


def enabled() -> bool:
    return bool(proxy_url())


def encode_trace() -> str | None:
    ctx = current()
    if ctx is None:
        return None
    raw = json.dumps(ctx.to_payload(), separators=(",", ":"), default=str)
    return base64.urlsafe_b64encode(raw.encode()).decode()


def decode_trace(value: str) -> dict[str, Any]:
    """Inverse of :func:`encode_trace` (used by the proxy)."""
    padded = value + "=" * (-len(value) % 4)
    return json.loads(base64.urlsafe_b64decode(padded.encode()))


def route(url: str) -> tuple[str, dict[str, str]]:
    """Where to send a request for *url*, and the headers to add.

    Returns *url* unchanged and no headers when tracing is off.
    """
    proxy = proxy_url()
    if not proxy:
        return url, {}
    parts = urlsplit(url)
    target = f"{proxy}{_FORWARD_PREFIX}{parts.path or '/'}"
    if parts.query:
        target += f"?{parts.query}"
    headers = {UPSTREAM_HEADER: f"{parts.scheme}://{parts.netloc}"}
    trace = encode_trace()
    if trace:
        headers[TRACE_HEADER] = trace
    return target, headers


def _reroute(request: httpx.Request) -> None:
    if request.headers.get(UPSTREAM_HEADER):
        return  # already routed (a retry re-sends the same request object)
    target, headers = route(str(request.url))
    if not headers:
        return
    request.url = httpx.URL(target)
    # httpx fixed Host when the request was built; it must name the proxy now.
    request.headers["Host"] = request.url.netloc.decode()
    request.headers.update(headers)


async def _reroute_async(request: httpx.Request) -> None:
    _reroute(request)


def sync_http_client(**kwargs: Any) -> httpx.Client | None:
    """An OpenAI-compatible httpx client that routes through the proxy.

    ``None`` when tracing is off, so callers keep their default client.
    """
    if not enabled():
        return None
    from openai import DefaultHttpxClient

    return DefaultHttpxClient(event_hooks={"request": [_reroute]}, **kwargs)


def async_http_client(**kwargs: Any) -> httpx.AsyncClient | None:
    """Async counterpart of :func:`sync_http_client`."""
    if not enabled():
        return None
    from openai import DefaultAsyncHttpxClient

    return DefaultAsyncHttpxClient(event_hooks={"request": [_reroute_async]}, **kwargs)
