# Auto-Ontology on all-Azure: effort estimate

## Context
The user wants to know how much work it would take to run NVIDIA/auto-ontology entirely on Azure: Azure OpenAI, Azure Database for PostgreSQL, Key Vault, Entra ID, and hosting. I read a shallow clone (~173k LOC: a Next.js frontend, a FastAPI backend, an ingestion worker and an MCP server). This file is the estimate and the plan of work. No code has been written.

## What the repo uses today, and how far each piece is from Azure
| Piece | Today | Distance to Azure |
|---|---|---|
| Chat LLM | `auto_ontology/utils/llm_invoke.py:_build_client`. Models whose names start with `openai/`, `azure/` or `aws/` go to `ChatOpenAI(base_url=…)`. Everything else goes to `ChatNVIDIA` | **Small.** The client is already OpenAI-compatible. However, the `azure/` prefix is sent to the endpoint as the model name, and Azure deployment names can't contain `/`. The prefix has to be stripped, or replaced with a provider flag |
| Embeddings | `auto_ontology/utils/embedding.py` uses the `openai.OpenAI` client, but sends NIM-only `extra_body` fields (`input_type`, `truncate`, `modality`) | **Small.** Send those fields only to NIM endpoints. Pass `dimensions=2048` to `text-embedding-3-large`, because the pgvector index is hard-coded to `halfvec(2048)` (`vdb/postgres.py`). Everything has to be re-ingested afterwards |
| Reranker | `auto_ontology/utils/rerank.py` makes a raw httpx call. Its `/rerank` branch already uses the Cohere-style schema (`results[].relevance_score`). It has one caller, `semantic/bridge_tables.py` | **Small.** Azure OpenAI has no reranker. Either point it at Cohere Rerank on Azure AI Foundry (likely no code change) or turn it off |
| Postgres | Built from `POSTGRES_*` env vars in `infra/postgres.py`, `frontend/lib/prisma.ts` and `frontend/prisma.config.ts`. Needs the `vector` and `pg_trgm` extensions | **Small with a password:** allow-list the extensions, add `sslmode=require` (Prisma/node-pg won't use TLS otherwise), URL-encode the password, and add a Helm switch to turn off the bundled postgres and pgadmin. **Medium with Entra/managed-identity auth:** about 15 direct `psycopg.connect` calls, a psycopg pool, the langchain_postgres `PGEngine` (SQLAlchemy), Alembic and Prisma would all need a token callback in place of a static password |
| Secrets | Env vars in a Helm Secret (`bundledSecret.enabled` lets you supply your own). Connection credentials go to HashiCorp Vault (`connectors/vault.py`, 153 LOC, 4 functions, 2 callers) or are stored **in plaintext in Postgres** | **Small:** load env secrets from Key Vault with the AKS CSI driver or Container Apps Key Vault references, with no code changes. **Medium:** add an Azure Key Vault backend that replaces `vault.py` |
| Auth | Better Auth 1.7 with the `@better-auth/sso` OIDC plugin (one provider, set up in the UI). Two roles: admin and viewer | **Very small for login:** Entra ID is a standard OIDC issuer; register `<APP_URL>/api/auth/sso/callback`. **Medium if you want Entra groups mapped to admin/viewer**, which isn't built in |
| Hosting | Docker Compose, or a Helm chart for Kubernetes | AKS works with the chart as is. Container Apps would need its own manifests or Bicep |

Neo4j appears only in comments; it isn't a runtime dependency. KumoRFM is optional and can stay off.

## Tier 1: runs on Azure, using keys and passwords (about 1–1.5 engineer-weeks)
1. **Azure OpenAI chat (0.5–1 day):** in `_build_client`, strip the provider prefix before sending the model name, or add an `LLM_PROVIDER=azure_openai` switch. Use the v1 endpoint `https://<res>.openai.azure.com/openai/v1/`. Check how `max_tokens` behaves on gpt-5 / o-series deployments.
2. **Embeddings (0.5–1 day):** send `extra_body` only when the endpoint is NIM, pass `dimensions=2048`, skip the NIM text prefix, then re-ingest.
3. **Reranker (0.5 day):** test Cohere Rerank on Azure AI Foundry against the existing `/rerank` branch, or turn it off.
4. **Azure Database for PostgreSQL Flexible Server (1 day):** set `azure.extensions=VECTOR,PG_TRGM`, add `sslmode=require` to the Python and Prisma connection strings, URL-encode the password, and add `postgres.enabled` / `pgadmin.enabled` switches to the Helm chart.
5. **Key Vault for env secrets (0.5–1 day):** set `bundledSecret.enabled=false` and add a SecretProviderClass (CSI driver) or External Secrets Operator.
6. **Entra ID login (0.5 day):** register an app, configure SSO with issuer `https://login.microsoftonline.com/<tenant>/v2.0`, and test. New users arrive as viewers, and an admin has to promote them by hand.
7. **Quality check (2–3 days, the biggest unknown):** the prompts are tuned for Nemotron and Claude. Run [ontology-sql-eval](https://github.com/NVIDIA/ontology-sql-eval) with GPT models and fix any structured-output or prompt regressions.

Infrastructure as code (AKS or Container Apps, ACR, PostgreSQL, Key Vault, Azure OpenAI, private endpoints and VNet) adds **2–5 days**, depending on your organization's standards.

## Tier 2: Azure-native and passwordless (another 1.5–2.5 engineer-weeks)
- **Azure OpenAI through managed identity (1–2 days):** use an Entra token provider (`azure-identity`) in place of the API key in all three model call paths: chat, embeddings and rerank.
- **PostgreSQL with Entra auth (2–4 days):** route every connection through one helper that fetches a fresh token. That covers the psycopg connects and pool, the SQLAlchemy `do_connect` event used by `PGEngine`, Alembic `env.py`, and Prisma (node-pg accepts an async `password` function).
- **Key Vault replacing HashiCorp Vault for connection credentials (1–2 days):** a new `connectors/azure_keyvault.py` with the same 4-function interface. Key Vault secret names allow only letters, digits and dashes, so database names must be sanitized. Choose the backend with an env flag.
- **Entra groups mapped to admin/viewer (1–2 days):** a Better Auth provisioning hook that reads the `groups` or `roles` claim.
- **AKS Workload Identity wiring and federated credentials (1 day).**

## Ongoing cost
The project does not accept external contributions (README). That means you would keep these changes in a fork and rebase on each upstream release. Keeping the changes behind env flags and in a few files (`llm_invoke.py`, `embedding.py`, `infra/postgres.py`, `vault.py`, the Helm chart) keeps rebases cheap.

## Verification (when the work is done)
- `docker compose` locally, pointed at real Azure OpenAI, Azure PostgreSQL and Key Vault endpoints. Then ingest Wide World Importers, turn on semantic compilation, and run the README's sample question.
- Run the existing pytest suites: `vdb/tests`, the connectors tests and the `mcp/tests`.
- Run ontology-sql-eval and compare against the Nemotron baseline.
- Entra login as two users in different groups and check the role each gets. Then rotate a Key Vault secret and check the app picks up the new value.
