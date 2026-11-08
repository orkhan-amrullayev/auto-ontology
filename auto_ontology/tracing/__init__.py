# SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
# SPDX-License-Identifier: Apache-2.0

"""LLM call tracing.

Off unless ``LLM_TRACE_PROXY_URL`` is set. When it is, every outbound model
request (chat, embeddings, rerank) is routed through the trace proxy
(:mod:`auto_ontology.llm_trace_proxy`) with ``X-AO-*`` headers describing who
made it and where in the pipeline it came from; the proxy forwards it to the
real endpoint and records request and response together with that context.

* :mod:`.context` -- the trace context, spans, and propagation helpers.
* :mod:`.transport` -- turns the context into headers and reroutes requests.
"""
