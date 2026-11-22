# GPT-powered RAG answers (`POST /explain`)

RecallIQ answers a consumer question with an AI-written summary that is **grounded only in
the official Government of Canada recall notices it retrieved**. Every AI answer passes a set
of deterministic checks first; if it fails any of them, or OpenAI is unavailable, the
deterministic template answer is shown instead, with a short explanation of why.

## Flow

```
consumer query
  -> retrieval (BM25 | Model A minilm | Model B minilm_finetuned | Model C minilm_hard_negative)
  -> deterministic identifier checks (UPC / model / lot / best-before)      ml/verification
  -> evidence selection: top results, conflicting identifiers set aside     ml/rag/pipeline.py
  -> evidence status: no_results | identifier_mismatch | evidence_found
       no_results / identifier_mismatch -> template (the LLM is never called)
       evidence_found + provider=openai  -> GPT answer                      ml/rag/llm.py
  -> output guards (all must pass, else template + fallback_reason)         ml/rag/guards.py
  -> response: answer + cited recall ids + verbatim evidence fields + official URLs + disclaimer
```

| Module | Responsibility |
|---|---|
| `ml/rag/pipeline.py` | Evidence selection, evidence status, template, LLM orchestration and fallback, `GenerationStats` |
| `ml/rag/llm.py` | `LLMConfig`, structured context, system prompt, OpenAI call (strict JSON schema), error mapping |
| `ml/rag/guards.py` | `check_grounded` and `validate_answer`: the deterministic acceptance checks |
| `backend/app/config.py` | Typed settings, with the key held as `SecretStr` and loaded from `backend/.env` |
| `backend/app/routers/search.py` | `POST /explain`: retrieval, identifier checks, DB fields, then `explain()` |
| `backend/app/llm_check.py` | CLI check of key and model availability (`python -m app.llm_check`) |
| `frontend/src/components/search/EvidenceSummary.tsx` | Answer, provenance label, fallback note, cited sources, usage |

## Safety rules and how they are enforced

| Rule | Enforcement |
|---|---|
| Use only retrieved evidence | Context contains only stored DB fields. Prompt rule 1. Guard: every code- or number-like token must appear in the evidence |
| Never fabricate product names, model numbers, UPCs, lot codes | Prompt rule 2 (quote names verbatim). Guards: code/number tokens and every double-quoted span must appear in the evidence |
| Never fabricate citations | URLs are **not sent** to the model. Cited ids must be evidence ids. The server attaches official URLs. Guard rejects any URL that is not an evidence URL |
| Never alter dates | Prompt rule 2 asks for dates as `YYYY-MM-DD` (prompt `rag-v2`). Guard: a date written in words ("October 1, 2012", "October 2012") is accepted only if it equals an evidence date (a month-year must match an evidence date's month) or is copied verbatim from a notice; otherwise the answer is rejected |
| Preserve deterministic identifier matching | Verdicts come from `ml/verification`, and the model only sees them. Guard: a "code matches" claim is rejected unless a cited notice has a deterministic MATCH |
| Never declare a product safe | Prompt rule 4. Guard rejects safe / not recalled / unaffected / "no recalls" sentences unless they are hedged ("does not mean it is safe"). The disclaimer is always returned |
| Insufficient evidence | No results or only conflicting notices: the LLM is skipped. The model can also set `evidence_sufficient=false`, and the UI then says the notices may not be about the product |
| Prompt injection | Query and records are sent as JSON *data* (they cannot break structure). Control and bidi characters are stripped and fields length-capped. The system prompt says to ignore embedded instructions. Guards reject instruction echoes and any non-evidence content |
| API unavailable | Timeouts, connection errors, 429, 5xx, auth, unknown model, refusal, truncation and malformed JSON all fall back to the template (still HTTP 200) with `fallback_reason` |
| Secrets | Key read only from env or `backend/.env` (`SecretStr`). Never logged, returned or sent to the browser. Tests strip it from the environment |

The guards are conservative. A false rejection costs one template answer. A false acceptance
could tell a consumer something the notice does not say.

**Measured false rejections (live, 2026-11-19, gpt-4o-mini, BM25, 4 real queries).**
With `rag-v2`: 2 of 6 answers accepted; the 4 rejections were all harmless wording:
- 1× a sentence start after `". \n\n"` was read as a capitalised name ("Since") — **fixed**
  in the guard (`_NAME_WORD` look-behind; regression test in `tests/rag/test_guards.py`).
- 2× "the details are **not listed**" matched the safety-claim pattern (meant for "your
  product is not listed"). Root cause: rag-v2 rule 8 itself said "if a field is missing, say
  it is not listed". **Fixed in the prompt** (`rag-v3`: say "the notice does not state" it);
  the safety guard is unchanged and still rejects both phrasings.
- 1× reason not captured: the backend did not configure logging, so uvicorn dropped the
  `recalliq.rag` line — **fixed** (see Observability).

With `rag-v3` the two previously rejected queries were both accepted (2 calls). No live
answer wrote a date in words (the dates seen were ISO); that path is covered by unit tests.
These numbers are anecdotal. One accepted answer added outside knowledge ("a serious
foodborne pathogen") that a lexical guard cannot detect — rating summaries by hand on
validation queries is still open (`CHECKPOINT.md` item 6).

## Configuration (`backend/.env`, see `backend/.env.example`)

| Variable | Default | Meaning |
|---|---|---|
| `RECALLIQ_LLM_PROVIDER` | `none` (example file: `openai`) | `openai` enables GPT answers. `none` means template only, with no network |
| `OPENAI_API_KEY` (or `RECALLIQ_OPENAI_API_KEY`) | (none) | Secret. Backend only |
| `RECALLIQ_OPENAI_MODEL` | `gpt-4o-mini` | Any chat model that supports Structured Outputs |
| `RECALLIQ_LLM_TIMEOUT_SECONDS` | `15` | Per-request timeout |
| `RECALLIQ_LLM_MAX_RETRIES` | `1` | SDK retries for 429 / 5xx / connection errors |
| `RECALLIQ_LLM_MAX_OUTPUT_TOKENS` | `500` | Output cap (`max_completion_tokens`) |
| `RECALLIQ_LLM_TEMPERATURE` | `0` | Sampling temperature |
| `RECALLIQ_RAG_TOP_K` | `5` | Results retrieved for `/explain` when the request omits `top_k` (1 to 5) |
| `RECALLIQ_RAG_MAX_EVIDENCE` | `3` | Notices passed to the model (1 to 5) |
| `RECALLIQ_LOG_LEVEL` | `INFO` | `recalliq.*` log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). `DEBUG` adds rejection excerpts, which may echo query content |

The frontend waits up to 45 s for `/explain` (`EXPLAIN_TIMEOUT_MS`). The summary is fetched
only when the user opens it, so a search never spends tokens by itself.

## Observability

Each response includes `explanation.generation`: `provider`, `model` (as reported by
OpenAI), `outcome` (`accepted | rejected | error:<code> | skipped:<status>`),
`prompt_version`, `latency_ms`, `prompt_tokens`, `completion_tokens` and `total_tokens`.
Tokens are recorded even when an answer is rejected. The same data is logged on the
`recalliq.rag` logger as one `rag_generation …` line per request, with the query length
only (never the query text or the key). The backend attaches a stderr handler to the
`recalliq` logger at startup (`app.main.configure_logging`), so these lines appear in the
uvicorn console. For a rejection or failure the INFO line carries only the fixed **reason
category** (e.g. `reason='unsupported safety claim'`). The excerpt that triggered it can
echo the consumer's input (a code they typed), so it is logged only at DEBUG
(`rag_generation reason_detail=…`, enabled with `RECALLIQ_LOG_LEVEL=DEBUG` while diagnosing).

## Checking the model

```bash
cd backend
python -m app.llm_check              # provider, model, key present?, model visible to this key?
python -m app.llm_check --generate   # + one real grounded answer on SYNTHETIC evidence
```

## Tests (OpenAI always mocked)

`tests/rag/test_llm.py` covers the pipeline and guards: grounding, citations, hallucinated
codes, names, URLs and ids, safety and identifier claims, missing evidence, every API failure
class, malformed or truncated output, prompt injection, config, and logging.
`tests/backend/test_explain_llm.py` covers `/explain` end to end with settings overrides.
An autouse fixture in `tests/conftest.py` removes OpenAI env vars and makes building a real
client fail, so the test suite can never call the paid API.

## Limitations

- The guards are lexical. They catch invented codes, names, URLs and ids, plus unhedged
  safety and identifier claims. They cannot prove that every paraphrased sentence is entailed
  by the evidence (for example, a wrong hazard described in different words).
- Output quality depends on the retrieved notices: retrieval errors propagate. The index is
  a 1,000-notice sample. Some stored `what_you_should_do` texts have list items joined
  without spaces (an artifact of the source data, left unmodified).
- The consumer query and evidence fields are sent to OpenAI when AI answers are enabled.
- The answer is English-only. Generation has not yet been evaluated on the annotated query
  sets (the held-out test evaluation has not been run).
