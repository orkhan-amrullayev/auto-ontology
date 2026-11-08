# SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
# SPDX-License-Identifier: Apache-2.0

"""Turn raw request/response bytes into what a trace row stores.

Pure functions, no I/O: classify the endpoint, parse bodies, drop what must not
or need not be stored (embedding vectors, oversized strings), reassemble a
streamed (SSE) chat completion, and pull out token usage.

Secrets never get here: they travel in headers (``Authorization``/``api-key``),
and the proxy stores no headers at all.
"""

from __future__ import annotations

import json
from typing import Any

#: Longest string kept in a stored body; longer ones are cut with a marker.
MAX_STRING_CHARS = 20_000


def classify(path: str) -> str:
    """``chat`` | ``embedding`` | ``rerank`` | ``other`` from the request path."""
    lowered = path.lower().rstrip("/")
    if lowered.endswith(("/chat/completions", "/completions", "/responses")):
        return "chat"
    if lowered.endswith("/embeddings"):
        return "embedding"
    if lowered.endswith(("/rerank", "/ranking", "/reranking")):
        return "rerank"
    return "other"


def parse_json(raw: bytes) -> Any:
    """*raw* as JSON, or ``{"_raw": <text>}`` when it is not JSON."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return {"_raw": _truncate(raw.decode("utf-8", errors="replace"))}


def _truncate(value: str) -> str:
    if len(value) <= MAX_STRING_CHARS:
        return value
    return (
        f"{value[:MAX_STRING_CHARS]}… [truncated {len(value) - MAX_STRING_CHARS} chars]"
    )


def _shrink(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, list):
        return [_shrink(item) for item in value]
    if isinstance(value, dict):
        return {key: _shrink(item) for key, item in value.items()}
    return value


def redact_response(endpoint: str, body: Any) -> Any:
    """*body* fit for storage: embedding vectors replaced by their size."""
    if endpoint == "embedding" and isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, list):
            body = {
                **body,
                "data": [
                    {
                        **{k: v for k, v in item.items() if k != "embedding"},
                        "embedding_dims": len(item.get("embedding") or []),
                    }
                    if isinstance(item, dict)
                    else item
                    for item in data
                ],
            }
    return _shrink(body)


def redact_request(body: Any) -> Any:
    return _shrink(body)


def usage(body: Any) -> tuple[int | None, int | None, int | None]:
    """``(prompt, completion, total)`` tokens reported in *body*, if any."""
    if not isinstance(body, dict):
        return None, None, None
    reported = body.get("usage")
    if not isinstance(reported, dict):
        return None, None, None
    prompt = reported.get("prompt_tokens", reported.get("input_tokens"))
    completion = reported.get("completion_tokens", reported.get("output_tokens"))
    total = reported.get("total_tokens")
    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)
    return prompt, completion, total


def assemble_stream(raw: bytes) -> dict[str, Any]:
    """Rebuild a chat completion from its server-sent-event stream.

    Concatenates the content deltas and tool-call argument deltas of choice 0,
    and keeps the last ``usage`` and ``model`` seen.
    """
    content: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    result: dict[str, Any] = {"streamed": True, "chunks": 0}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        result["chunks"] += 1
        if chunk.get("model"):
            result["model"] = chunk["model"]
        if isinstance(chunk.get("usage"), dict):
            result["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for call in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(
                    call.get("index", 0), {"name": "", "arguments": ""}
                )
                function = call.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                result["finish_reason"] = choice["finish_reason"]
    result["content"] = _truncate("".join(content))
    if tool_calls:
        result["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return result
