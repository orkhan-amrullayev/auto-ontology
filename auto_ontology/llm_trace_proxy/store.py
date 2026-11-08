# SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
# SPDX-License-Identifier: Apache-2.0

"""Write trace rows to ``llm_trace_call``. Best-effort: a failed write is
logged and dropped, never surfaced to the model call it describes."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import insert

from auto_ontology.dal.schema import llm_trace_call
from auto_ontology.dal.session import get_engine

logger = logging.getLogger(__name__)


def record_call(row: dict[str, Any]) -> None:
    try:
        with get_engine().begin() as conn:
            conn.execute(insert(llm_trace_call).values(**row))
    except Exception:  # noqa: BLE001 — tracing must not break model calls
        logger.exception(
            "Failed to record LLM trace call (trace=%s)", row.get("trace_id")
        )
