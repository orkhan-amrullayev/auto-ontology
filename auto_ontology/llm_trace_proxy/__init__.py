# SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
# SPDX-License-Identifier: Apache-2.0

"""Reverse proxy that records every model request with its trace context.

The app routes model requests here when ``LLM_TRACE_PROXY_URL`` is set (see
:mod:`auto_ontology.tracing.transport`). Each request is forwarded unchanged to
the origin named by ``X-AO-Upstream`` -- only origins of the configured model
endpoints are allowed -- and the request, response, timing, token usage and the
``X-AO-Trace`` context are written to the ``llm_trace_call`` table.

Run with ``python -m auto_ontology.llm_trace_proxy`` (port
``LLM_TRACE_PROXY_PORT``, default 3004).
"""
