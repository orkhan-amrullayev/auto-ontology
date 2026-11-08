# SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
# SPDX-License-Identifier: Apache-2.0

"""The trace context carried alongside a unit of work.

A *trace* is one unit of work worth looking at as a whole: a chat turn, a
semantic compilation of one database, an ingest, an API request. It carries
who started it (user, conversation, turn) and a *span path*: the stack of
pipeline steps the code is currently inside (``chat › construct_sql ›
SQLGenerationModel``). Every model call made inside it is stamped with a copy,
so the proxy can stitch calls into a tree.

It lives in a :class:`~contextvars.ContextVar`. That follows ``asyncio`` tasks,
``asyncio.to_thread`` and LangGraph nodes on its own, but not:

* plain ``ThreadPoolExecutor``/``threading.Thread`` workers -- use
  :class:`TracingThreadPoolExecutor` or :func:`bind`, which carry *only* this
  variable (``dal.session`` keeps its transaction in another context variable
  that worker threads must not inherit, so ``copy_context()`` is not safe);
* the chat worker *process* -- serialise with :meth:`TraceContext.to_payload`
  and restore with :func:`activate`.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class Span:
    id: str
    name: str


@dataclass(frozen=True)
class TraceContext:
    """Who and where. Immutable: entering a span makes a new one."""

    trace_id: str
    #: What kind of work this is: ``chat``, ``semantic_compilation``,
    #: ``ingest``, ``api`` ...
    kind: str
    user_id: str | None = None
    conversation_id: str | None = None
    #: The chat turn (``frontend.conversation_analytics.id``) when there is one.
    turn_id: str | None = None
    source: str | None = None
    database_name: str | None = None
    #: A short human label for the trace list (e.g. the question asked).
    title: str | None = None
    spans: tuple[Span, ...] = ()
    #: Details of the model call about to be made (schema name, wrapper,
    #: attempt...). Set just around a single call by :func:`call_metadata`.
    call: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """A picklable/JSON-able form, for handing to another process."""
        payload = asdict(self)
        payload["spans"] = [asdict(span) for span in self.spans]
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TraceContext:
        data = dict(payload)
        data["spans"] = tuple(Span(**span) for span in data.get("spans") or ())
        return cls(**data)


def restore(payload: dict[str, Any] | None) -> TraceContext | None:
    """:meth:`TraceContext.from_payload`, or ``None`` for a missing or bad one.

    Tracing must never break the work it observes.
    """
    if not payload:
        return None
    try:
        return TraceContext.from_payload(payload)
    except Exception:  # noqa: BLE001
        return None


_current: ContextVar[TraceContext | None] = ContextVar("ao_trace", default=None)


def current() -> TraceContext | None:
    return _current.get()


def _new_id() -> str:
    return uuid.uuid4().hex


@contextlib.contextmanager
def activate(ctx: TraceContext | None) -> Iterator[TraceContext | None]:
    """Make *ctx* the current trace for the duration of the block."""
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


@contextlib.contextmanager
def start_trace(
    kind: str, *, trace_id: str | None = None, **fields: Any
) -> Iterator[TraceContext]:
    """Begin a new trace (replacing any current one) for the block."""
    ctx = TraceContext(trace_id=trace_id or _new_id(), kind=kind, **fields)
    with activate(ctx):
        yield ctx


@contextlib.contextmanager
def ensure_trace(kind: str, **fields: Any) -> Iterator[TraceContext]:
    """Join the current trace, or start one when there is none."""
    ctx = current()
    if ctx is not None:
        yield ctx
        return
    with start_trace(kind, **fields) as started:
        yield started


@contextlib.contextmanager
def span(name: str) -> Iterator[None]:
    """Enter a named pipeline step. A no-op outside a trace."""
    ctx = current()
    if ctx is None:
        yield
        return
    with activate(replace(ctx, spans=(*ctx.spans, Span(_new_id(), name)))):
        yield


@contextlib.contextmanager
def call_metadata(**meta: Any) -> Iterator[None]:
    """Describe the model call(s) made inside the block. No-op outside a trace."""
    ctx = current()
    if ctx is None:
        yield
        return
    with activate(replace(ctx, call={**ctx.call, **meta})):
        yield


def bind(fn: Callable[P, R]) -> Callable[P, R]:
    """*fn*, but run under the trace that is current *now*.

    For handing work to threads that do not inherit context. Only the trace
    variable is carried; nothing else from the caller's context is.
    """
    ctx = current()
    if ctx is None:
        return fn

    def bound(*args: P.args, **kwargs: P.kwargs) -> R:
        with activate(ctx):
            return fn(*args, **kwargs)

    return bound


class TracingThreadPoolExecutor(ThreadPoolExecutor):
    """A ``ThreadPoolExecutor`` whose tasks run under the submitter's trace.

    ``map`` goes through ``submit``, so it is covered too.
    """

    def submit(self, fn: Callable[..., R], /, *args: Any, **kwargs: Any) -> Future[R]:
        return super().submit(bind(fn), *args, **kwargs)


class TraceRequestsMiddleware:
    """ASGI middleware: run each HTTP request inside an ``api`` trace.

    Gives model calls made directly by a request handler (description
    suggestions, entity coverage...) a trace with the caller and route. Routes
    that start a trace of their own (chat) replace it for their work. Requests
    that make no model call leave nothing behind.
    """

    def __init__(self, app: Any, *, user_header: str) -> None:
        self.app = app
        self._user_header = user_header.lower().encode()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        user_id = headers.get(self._user_header, b"").decode().strip() or None
        source = headers.get(b"x-auto-ontology-source", b"").decode().strip() or None
        with start_trace(
            "api",
            user_id=user_id,
            source=source,
            title=f"{scope.get('method', '')} {scope.get('path', '')}",
        ):
            await self.app(scope, receive, send)
