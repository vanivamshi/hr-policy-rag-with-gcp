# HR Policy RAG Assistant on Google Cloud

An employee-facing chat assistant that answers questions from company HR policy documents,
with citations, safety guardrails, a semantic cache, LLM fallback and tracing/evaluation.

```
Layer 1  Browser ─► Google OAuth (st.login) + employee allow-list ◄── Secret Manager (streamlit-auth)
Layer 2  Cloud Run "hr-rag-assistant":
         Streamlit UI ─► Input guardrail ─► Semantic cache ──hit──────────────┐
                         (Model Armor)        │ miss                          ▼
                                              └─► Guarded agent ─► Output guardrail ─► UI
                                                  ├ guarded search tool    (Model Armor)
                                                  └ LiteLLM router
Layer 3  GCS ─(offline ingest)─► Qdrant Cloud (dense + BM25)  ·  Jina (embeddings + reranker)
         Vertex AI Gemini (primary) / Groq (fallback)  ·  LangSmith (traces + eval)
```

## Code map

| Component (diagram)        | Code |
|----------------------------|------|
| Google OAuth + allow-list  | [app.py](app.py), [src/hr_rag/auth.py](src/hr_rag/auth.py) |
| Input/Output guardrails    | [src/hr_rag/guardrails.py](src/hr_rag/guardrails.py) (Model Armor) |
| Semantic cache             | [src/hr_rag/cache.py](src/hr_rag/cache.py) (Qdrant collection `hr_semantic_cache`) |
| Guarded agent              | [src/hr_rag/agent/agent.py](src/hr_rag/agent/agent.py) |
| Guarded search tool        | [src/hr_rag/agent/tools.py](src/hr_rag/agent/tools.py) |
| LiteLLM router             | [src/hr_rag/llm.py](src/hr_rag/llm.py) |
| Hybrid retrieval + rerank  | [src/hr_rag/retrieval/](src/hr_rag/retrieval/) |
| Ingestion GCS → Qdrant     | [src/hr_rag/ingestion/](src/hr_rag/ingestion/) |
| Request flow               | [src/hr_rag/pipeline.py](src/hr_rag/pipeline.py) |
| LangSmith evaluation       | [eval/run_eval.py](eval/run_eval.py) |
| Deploy                     | [Dockerfile](Dockerfile), [scripts/](scripts/) |

### How a question is answered

1. **Input guardrail**: Model Armor `sanitizeUserPrompt` checks for prompt injection or jailbreak attempts, harmful content and malicious URLs. If Model Armor is unreachable, the request is blocked (fail-closed). Set `GUARDRAIL_FAIL_OPEN=true` to allow requests through instead.
2. **Condense**: a follow-up like "and for part-timers?" is rewritten as a standalone question, so the cache and retriever both work on multi-turn chats.
3. **Semantic cache**: a cached answer is returned if a previous question has cosine similarity of at least 0.95 and is within the TTL (7 days by default). A hit skips the agent entirely.
4. **Agent**: runs a tool-calling loop. It must call `search_hr_policies` on its first step, and may search again up to `AGENT_MAX_STEPS`. The tool does the following:
   - Runs a hybrid Qdrant query: Jina dense vectors plus BM25 sparse vectors, fused with RRF.
   - Reranks the results with the Jina reranker and drops anything under `RERANK_MIN_SCORE`.
   - Returns numbered excerpts wrapped in `<policy_excerpt>` tags, which the model is told to treat as data, not instructions.
5. **Citations**: `[n]` markers in the answer are matched to real retrieved chunks. Any number the model invents is dropped.
6. **Output guardrail**: Model Armor `sanitizeModelResponse` runs on the answer. If the only issue is sensitive data and Model Armor has returned a de-identified version, that version is shown. Otherwise the answer is blocked.
7. Only answers that are grounded, cited and safe are written to the cache.

LLM calls go through the LiteLLM Router. The primary model is `vertex_ai/gemini-2.5-flash`. If it errors or rate-limits (after retries), the router falls back to `groq/llama-3.3-70b-versatile`. The fallback only exists if `GROQ_API_KEY` is set.

## Local development

```bash
uv venv && uv pip install -e ".[dev]"          # or: python -m venv .venv && pip install -e ".[dev]"
cp .env.example .env                           # fill in Qdrant, Jina, Groq, LangSmith keys
gcloud auth application-default login          # Vertex AI + Model Armor + GCS use ADC

# Ingest: from a local folder, or from GCS
python -m hr_rag.ingestion.ingest --local-dir ./sample_docs
python -m hr_rag.ingestion.ingest --bucket $GCS_BUCKET --prefix hr-policies/

# Run the UI without Google login (dev only)
AUTH_DISABLED=true streamlit run app.py

# Or with real Google login
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in OAuth client
streamlit run app.py

pytest                                          # unit tests, no cloud calls
```

Supported formats are PDF, DOCX and PPTX. PDF and PPTX chunks keep their page or slide numbers for citations. Re-running ingestion does the following:
- Replaces each document's chunks. Chunk IDs are deterministic, and the old chunks for each `source` are deleted first.
- Clears the semantic cache, so no stale answers survive.

Scanned (image-only) PDFs have no text layer. They are skipped with a warning, so run OCR on them first.

## Deploy to Cloud Run

1. **Create a Google OAuth client.** In Cloud Console, go to APIs & Services → Credentials → OAuth client ID, and choose the "Web application" type. Fill in `.streamlit/secrets.toml`. Set `redirect_uri` after the first deploy (step 4).
2. **One-time setup.** This enables the APIs and creates the bucket, service account, IAM roles, secrets and Model Armor template:
   ```bash
   export PROJECT_ID=my-project BUCKET=my-hr-policies REGION=us-central1
   export JINA_API_KEY=... QDRANT_API_KEY=... GROQ_API_KEY=... LANGSMITH_API_KEY=...
   ./scripts/setup_gcp.sh
   ```
3. **Upload the policies and ingest them** (offline, one-time):
   ```bash
   gcloud storage cp ./policies/* gs://$BUCKET/hr-policies/
   python -m hr_rag.ingestion.ingest --bucket $BUCKET
   ```
4. **Deploy:**
   ```bash
   cp deploy/env.example.yaml deploy/env.yaml   # set QDRANT_URL, ALLOWED_DOMAINS, ...
   ./scripts/deploy.sh
   ```
   The script prints the service URL. Then:
   - Set `redirect_uri = "<URL>/oauth2callback"` in `secrets.toml`.
   - Add that same URI to the OAuth client.
   - Re-run `setup_gcp.sh` (it adds a new `streamlit-auth` secret version).
   - Re-run `deploy.sh`.

What deploy does:
- Secrets are injected from Secret Manager. `streamlit-auth` is mounted as a file at `/app/.streamlit/secrets.toml`, and the API keys are set as environment variables.
- The service is deployed with `--allow-unauthenticated`, because the app enforces Google login and the allow-list itself.
- Session affinity keeps each user's Streamlit websocket on the same instance.

## Evaluation (LangSmith)

Edit [eval/dataset.jsonl](eval/dataset.jsonl) and replace the `REPLACE` reference answers with the correct answers from your own policies. The dataset also includes out-of-scope and injection questions. Then run:

```bash
python eval/run_eval.py --experiment-prefix gemini-2.5-flash
```

This creates the LangSmith dataset once. It then runs every question through the full pipeline with the cache off, and scores each answer:

- **correctness**: LLM judge, compared against the reference answer
- **groundedness**: LLM judge, checking the answer against the retrieved excerpts
- **has_citation**: a simple check for whether the answer cites anything

You can compare experiments in the LangSmith UI. With `LANGSMITH_TRACING=true`, every production request is also traced: condense, cache, agent steps, LLM calls and guardrails. User emails are hashed before they reach the logs.

## Configuration

Every setting is an environment variable. See [src/hr_rag/config.py](src/hr_rag/config.py) for the full list and defaults. The main tuning knobs are:
- Retrieval: `RETRIEVAL_PREFETCH_K`, `RERANK_TOP_N`, `RERANK_MIN_SCORE`
- Cache: `CACHE_SIMILARITY_THRESHOLD`, `CACHE_TTL_SECONDS`
- Chunking: `CHUNK_SIZE`, `CHUNK_OVERLAP`
- Models: `PRIMARY_MODEL`, `FALLBACK_MODEL`

If you change `JINA_EMBEDDING_DIM`, run ingestion again with `--recreate`, and delete the cache collection.
