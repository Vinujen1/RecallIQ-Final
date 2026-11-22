# frontend/ — RecallIQ web app

Next.js 16 (App Router, `cacheComponents`), React 19, TypeScript and Tailwind CSS 4. The UI
holds no business logic and no secrets. Every result, identifier verdict and summary comes
from the FastAPI backend. The visual system and trust rules are specified in
[`DESIGN.md`](DESIGN.md).

## Run

```bash
npm install
cp .env.example .env.local      # NEXT_PUBLIC_API_BASE_URL, default http://localhost:8000
npm run dev                     # http://localhost:3000 (backend must be running)
```

`NEXT_PUBLIC_*` values are embedded in the browser bundle. Never put an API key there: the
OpenAI key belongs in `backend/.env` only.

## Routes

| Route | Purpose |
|---|---|
| `/` | Hero search and an explanation of what RecallIQ can and cannot tell you |
| `/search` | Results with URL-held state (query, model, domain, category, selected notice), identifier comparison, the on-demand evidence summary (`POST /explain`), and loading / empty / error / 503 states |
| `/recalls/[id]` | Server-rendered notice page; re-runs the originating search for the identifier comparison |
| `/model-lab` | Learning curves, proxy metrics with bootstrap CIs, live four-model comparison |
| `/how-it-works` | Method, limitations, FAQ |

## Layout

| Path | Contents |
|---|---|
| `src/app/` | Routes, layout, global styles (design tokens), error and not-found pages |
| `src/components/search/` | `SearchApp` (state and orchestration), result rows, detail panel, refine controls, identifier checks, `EvidenceSummary` (RAG answer and citations), `use-api` (cancellable fetch hook) |
| `src/components/lab/` | Model Lab charts and live comparison |
| `src/components/{brand,layout,recall,ui}/` | Wordmark, header/footer/theme, notice identifier panel, shared primitives |
| `src/lib/api/` | Typed API client (`client.ts`: timeouts, error kinds, no server detail leaked) and response types (`types.ts`, mirroring `backend/app/schemas.py`) |
| `src/lib/` | Model metadata, search-param parsing, formatting, navigation |
| `src/data/model-lab.json` | **Generated** by `node scripts/export-model-lab.mjs` from `../experiments/`. Never edited by hand |
| `src/proxy.ts` | Returns a real HTTP 404 for unknown recall IDs before the streamed page starts (Next 16 Proxy) |
| `tests/unit/`, `tests/e2e/` | Playwright: pure helpers; user journeys, API failure states, mobile layout |

## Checks

```bash
npx tsc --noEmit
npm run lint
npm run build
npm run test:unit
npm run test:e2e        # desktop + mobile; needs the backend on :8000 (or set E2E_BASE_URL)
```

`AGENTS.md` is written by `next dev` itself (agent notes about this Next.js version) and is
kept as generated.
