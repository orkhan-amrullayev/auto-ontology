# SPDX-FileCopyrightText: Copyright (c) 2026, Orkhan Amrullayev.
# SPDX-License-Identifier: Apache-2.0

"""``python -m auto_ontology.llm_trace_proxy`` -- serve the trace proxy."""

from __future__ import annotations

import logging
import os

import uvicorn

from auto_ontology.llm_trace_proxy.app import create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    uvicorn.run(
        create_app(),
        host=os.environ.get("LLM_TRACE_PROXY_HOST", "0.0.0.0"),
        port=int(os.environ.get("LLM_TRACE_PROXY_PORT", "3004")),
        # Request bodies hold prompts; uvicorn's access log only has paths,
        # which is fine, but keep it quiet like the other services.
        access_log=False,
    )


if __name__ == "__main__":
    main()
