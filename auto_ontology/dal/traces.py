# SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
# SPDX-License-Identifier: Apache-2.0

"""Read LLM traces recorded by the trace proxy (``llm_trace_call``).

A *trace* is the set of calls sharing a ``trace_id``. A call recorded without
one (made outside any traced work) is shown as a trace of its own, addressed as
``call:<id>``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text

from auto_ontology.dal.session import get_engine
from auto_ontology.infra.postgres import FRONTEND_SCHEMA

_UNTRACED_PREFIX = "call:"

# The trace key: the recorded trace id, or a per-call key for untraced calls.
_KEY = f"COALESCE(c.trace_id, '{_UNTRACED_PREFIX}' || c.id::text)"


def list_traces(
    *,
    limit: int = 50,
    before: datetime | None = None,
    kind: str | None = None,
    user_id: str | None = None,
    conversation_id: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """Newest traces first, one summary row per trace.

    ``before`` pages backwards: pass the ``started_at`` of the last row seen.
    ``search`` matches the title, the trace id, the user's email or a model.
    """
    filters = ["TRUE"]
    params: dict[str, Any] = {"limit": limit}
    if kind:
        filters.append("c.kind = :kind")
        params["kind"] = kind
    if user_id:
        filters.append("c.user_id = :user_id")
        params["user_id"] = user_id
    if conversation_id:
        filters.append("c.conversation_id = :conversation_id")
        params["conversation_id"] = conversation_id
    having = ["TRUE"]
    if before:
        having.append("MIN(c.started_at) < :before")
        params["before"] = before
    if search:
        having.append(
            "(bool_or(c.title ILIKE :search) OR "
            f"{_KEY} ILIKE :search OR "
            "bool_or(u.email ILIKE :search) OR bool_or(c.model ILIKE :search))"
        )
        params["search"] = f"%{search}%"

    sql = f"""
        SELECT
            {_KEY} AS trace_id,
            (array_agg(c.kind ORDER BY c.id) FILTER (WHERE c.kind IS NOT NULL))[1] AS kind,
            (array_agg(c.title ORDER BY c.id) FILTER (WHERE c.title IS NOT NULL))[1] AS title,
            (array_agg(c.user_id ORDER BY c.id) FILTER (WHERE c.user_id IS NOT NULL))[1] AS user_id,
            (array_agg(u.email ORDER BY c.id) FILTER (WHERE u.email IS NOT NULL))[1] AS user_email,
            (array_agg(u.name ORDER BY c.id) FILTER (WHERE u.name IS NOT NULL))[1] AS user_name,
            (array_agg(c.conversation_id ORDER BY c.id)
                FILTER (WHERE c.conversation_id IS NOT NULL))[1] AS conversation_id,
            (array_agg(c.source ORDER BY c.id) FILTER (WHERE c.source IS NOT NULL))[1] AS source,
            (array_agg(c.database_name ORDER BY c.id)
                FILTER (WHERE c.database_name IS NOT NULL))[1] AS database_name,
            MIN(c.started_at) AS started_at,
            MAX(c.started_at + make_interval(secs => COALESCE(c.duration_ms, 0) / 1000.0))
                AS ended_at,
            COUNT(*) AS calls,
            COUNT(*) FILTER (WHERE c.endpoint = 'chat') AS llm_calls,
            COUNT(*) FILTER (WHERE c.endpoint = 'embedding') AS embedding_calls,
            COUNT(*) FILTER (
                WHERE c.error IS NOT NULL OR c.status_code IS NULL OR c.status_code >= 400
            ) AS errors,
            COALESCE(SUM(c.prompt_tokens), 0) AS prompt_tokens,
            COALESCE(SUM(c.completion_tokens), 0) AS completion_tokens,
            COALESCE(SUM(c.total_tokens), 0) AS total_tokens,
            array_remove(array_agg(DISTINCT c.model), NULL) AS models
        FROM llm_trace_call c
        LEFT JOIN {FRONTEND_SCHEMA}."user" u ON u.id = c.user_id
        WHERE {" AND ".join(filters)}
        GROUP BY {_KEY}
        HAVING {" AND ".join(having)}
        ORDER BY MIN(c.started_at) DESC
        LIMIT :limit
    """
    with get_engine().connect() as conn:
        return [dict(row._mapping) for row in conn.execute(text(sql), params)]


def get_trace_calls(trace_id: str) -> list[dict[str, Any]]:
    """Every call of one trace, oldest first, with full bodies."""
    if trace_id.startswith(_UNTRACED_PREFIX):
        condition = "c.trace_id IS NULL AND c.id = :call_id"
        try:
            params: dict[str, Any] = {"call_id": int(trace_id[len(_UNTRACED_PREFIX) :])}
        except ValueError:
            return []
    else:
        condition = "c.trace_id = :trace_id"
        params = {"trace_id": trace_id}
    sql = f"""
        SELECT c.*, u.email AS user_email, u.name AS user_name
        FROM llm_trace_call c
        LEFT JOIN {FRONTEND_SCHEMA}."user" u ON u.id = c.user_id
        WHERE {condition}
        ORDER BY c.started_at, c.id
    """
    with get_engine().connect() as conn:
        return [dict(row._mapping) for row in conn.execute(text(sql), params)]
