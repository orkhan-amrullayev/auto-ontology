# SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
# SPDX-License-Identifier: Apache-2.0

"""API routes for the LLM Tracing page.

Read-only views over ``llm_trace_call``, which the trace proxy fills when LLM
tracing is enabled (see ``auto_ontology/tracing``). Admin-only: the Next.js
gateway requires the ``trace:read`` permission before forwarding here.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from auto_ontology.dal import traces as traces_dal
from auto_ontology.server.responses import TraceDetailResponse, TraceListResponse
from auto_ontology.tracing import transport

router = APIRouter()


@router.get("/traces", response_model=TraceListResponse)
def list_traces(
    limit: int = Query(50, ge=1, le=200),
    before: datetime | None = None,
    kind: str | None = None,
    user_id: str | None = None,
    conversation_id: str | None = None,
    search: str | None = Query(None, max_length=200),
) -> dict:
    """Recorded traces, newest first: who started each, what it was, how many
    model calls it made, tokens used and errors. Page with ``before`` (the
    ``started_at`` of the last trace already shown). ``enabled`` says whether
    this backend is currently routing model calls through the trace proxy.
    """
    return {
        "enabled": transport.enabled(),
        "items": traces_dal.list_traces(
            limit=limit,
            before=before,
            kind=kind,
            user_id=user_id,
            conversation_id=conversation_id,
            search=search,
        ),
    }


@router.get("/traces/{trace_id}", response_model=TraceDetailResponse)
def get_trace(trace_id: str) -> dict:
    """Every model call of one trace, oldest first, with its pipeline position
    (``spans``), call details (schema, structured-output method, attempt),
    request and response bodies, timing and token usage."""
    calls = traces_dal.get_trace_calls(trace_id)
    if not calls:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"trace_id": trace_id, "calls": calls}
