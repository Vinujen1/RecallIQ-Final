# RecallIQ — Website Design Specification

**Direction:** Futuristic Champagne Gold
**Brand:** RecallIQ
**Status:** v2.1 approved 2026-11-16 and implemented (Phase 5). Owner decisions: backend filters (Option A); temporary wordmark approved; label-scanner hero; **BM25 default model**; headline "Know what you own. Find what matters." Where §6 says Model C is the default or shows the working headline, these decisions win.
**Stack (preserved):** Next.js 16.4 (App Router, `cacheComponents`), React 19.3, Tailwind CSS 4.3, TypeScript. No new runtime dependencies are planned.

---

## 0. Product frame

RecallIQ helps a person in Canada answer one question: **"Is there an official recall notice that matches the product I have?"** The answer is always a pointer to the Government of Canada notice, never a verdict about the user's product.

| User | Job | Primary route |
|---|---|---|
| Consumer holding a product | Describe it in plain words, find matching official notices, check the codes on the package | `/` → `/search` → `/recalls/[id]` |
| Course assessor / technical reviewer | Understand the four retrieval models and judge the evidence | `/model-lab`, `/how-it-works` |
| Team member demoing at the checkpoint | Run live queries across models, show identifier checks and grounded explanations | `/search` |

**Trust rules (apply to every screen):**
1. Ranked rows are **"closest notices in the index"**, never "matches". Retrieval has no relevance threshold, and the semantic models always return `top_k` rows. "Possible match" is reserved for rows with an identifier MATCH. No result never means "safe".
2. Similarity scores are never shown as percentages or probabilities. Ranks are primary. A raw score appears only in a labelled technical disclosure, beside its `score_type`. BM25 and cosine scores are never placed on a shared scale.
3. Identifier verdicts are shown exactly as returned. Their emphasis follows **what they mean for the user**, not good/bad colour coding (see §2.2):
   - MATCH: your code appears in the notice, so act.
   - MISMATCH: your code is not listed.
   - UNKNOWN: the codes could not be compared.
4. The official notice link is the most prominent action on every record. A link to the official recalls search is visible in every search state.
5. Coverage is always stated where results appear: "Searches a 1,000-notice sample of Government of Canada recalls — not all recalls."
6. Proxy metrics (generated validation queries) are labelled as such wherever they appear.
7. Fields RecallIQ derives itself (e.g. `product_family`) are labelled "RecallIQ grouping — not from the notice" or are not shown.

## 1. Brand identity

**Logo status: the selected RecallIQ logo asset is NOT in the repository** (searched for `*logo*`, `*brand*`, `*wordmark*`; none were found). Until it is supplied:
- Use a **temporary typographic wordmark**: "Recall" in Manrope 600 and "IQ" in Manrope 700 Champagne Gold, with no invented symbol.
- The component is `components/brand/Wordmark.tsx` and carries `data-temporary-logo="true"` plus a `TODO(logo)` comment.
- The favicon stays the default until the real asset arrives.
- Once the asset arrives, place the SVG in `public/brand/` (full, icon-only, dark and light variants). Only the `Wordmark` component changes.

Personality: precise, calm, editorial, quietly technical. It reads like a well-made instrument, not a dashboard or a chatbot.

## 2. Colour system

### 2.1 Raw palette

| Name | Hex | Role |
|---|---|---|
| Obsidian | `#101112` | dark background |
| Graphite | `#202124` | dark elevated surface |
| Champagne Gold | `#D9B98A` | brand accent; dark-mode action |
| Warm Ivory | `#F8F4EA` | light background; dark-mode primary text |
| Soft Stone | `#B9B2A8` | dark-mode secondary text |
| Deep Ink | `#25231F` | light-mode primary text |
| *Bronze (derived)* | `#7A5C32` | light-mode gold for text and links (Champagne on Ivory is only 1.7:1) |
| *Ash (derived)* | `#9C968D` / `#6B655C` | muted text, dark / light |

### 2.2 Semantic tokens

These are CSS custom properties on `:root[data-theme]`, exposed to Tailwind 4 through `@theme inline`.

| Token | Dark | Light | Use |
|---|---|---|---|
| `--bg` | `#101112` | `#F8F4EA` | page |
| `--surface` | `#17181A` | `#FFFFFF` | panels, result list |
| `--surface-raised` | `#202124` | `#FFFDF8` | detail panel, menus, sheet |
| `--text` | `#F8F4EA` | `#25231F` | primary text |
| `--text-secondary` | `#B9B2A8` | `#5F5A52` | body copy, metadata |
| `--text-muted` | `#9C968D` | `#6B655C` | captions, hints |
| `--accent` | `#D9B98A` | `#7A5C32` | links, focus, active nav, key numerals |
| `--accent-fill` | `#D9B98A` | `#D9B98A` | primary button fill |
| `--on-accent` | `#101112` | `#101112` | text on accent fill |
| `--border` | `rgb(248 244 234 / 0.10)` | `rgb(37 35 31 / 0.12)` | hairlines |
| `--border-strong` | `rgb(248 244 234 / 0.20)` | `rgb(37 35 31 / 0.24)` | inputs, selected rows |
| `--focus` | `#D9B98A` | `#7A5C32` | two-tone ring: 2px `--bg` gap and a 2px `--focus` outline (stays visible on the gold button) |
| `--alert` | `#F5A35C` | `#9A4A0C` | identifier **MATCH** (code appears in a recall notice, so act). Strongest emphasis, distinct from brand gold |
| `--neutral` | `#B9B2A8` | `#5F5A52` | identifier **MISMATCH** (calm: code not listed) |
| `--unknown` | `#9C968D` | `#6B655C` | identifier **UNKNOWN** (neutral) |
| `--danger` | `#F08A7E` | `#A63A2E` | **errors only** (API down, 503, validation). Never a verdict |
| `--info` | `#8FB4E8` | `#2F5E9E` | proxy-metric labels, coverage and notice callouts |

No verdict uses green: a green "match" would read as "safe" when it means the opposite. No verdict shares a token with error states.

### 2.3 Measured contrast

These ratios were computed with the WCAG 2.x relative-luminance formula on 2026-11-16. They are design-time values and will be re-checked in the browser.

| Pair | Ratio |
|---|---|
| Ivory on Obsidian | 17.2 |
| Gold on Obsidian | 10.1 |
| Gold on Graphite | 8.6 |
| Stone on Obsidian | 9.0 |
| Stone on Graphite | 7.7 |
| Ash on Obsidian | 6.5 |
| Ink on Ivory | 14.3 |
| Bronze on Ivory | 5.6 |
| Bronze on White | 6.2 |
| `#5F5A52` on Ivory | 6.2 |
| `#6B655C` on Ivory | 5.3 |
| Obsidian on Gold (button) | 10.1 |
| alert / danger / info, dark (on Obsidian) | 9.3 / 7.8 / 8.9 |
| alert / danger / info, light (on Ivory) | 5.7 / 5.9 / 6.0 |
| alert on Graphite / alert on White | 7.9 / 6.3 |

**Champagne Gold is never used as text on light backgrounds.** Status is always shown as icon + words, never colour alone.

### 2.4 Themes

- Dark is the brand default; light is a full theme, not an inversion.
- `data-theme` on `<html>` is set before paint by a tiny inline script: the stored choice, else `prefers-color-scheme`. This avoids a flash.
- The theme toggle offers System, Dark and Light and persists to `localStorage` (guarded with try/catch).

## 3. Typography

All fonts load through `next/font/google` (self-hosted at build; no runtime request to Google): Manrope (variable), Geist, Geist Mono.

| Style | Font | Size (mobile → desktop) | Weight | Line height | Tracking |
|---|---|---|---|---|---|
| Display | Manrope | `clamp(2.5rem, 6vw, 4.75rem)` | 600 | 1.02 | −0.035em |
| H1 | Manrope | `clamp(2rem, 4vw, 3rem)` | 600 | 1.1 | −0.025em |
| H2 | Manrope | `clamp(1.5rem, 2.6vw, 2.125rem)` | 600 | 1.15 | −0.02em |
| H3 | Manrope | 1.25rem | 600 | 1.3 | −0.01em |
| Eyebrow (implemented as `.kicker`) | Geist | 0.875rem | 500 | 1.4 | +0.01em, sentence case (changed in implementation: tracked all-caps mono eyebrows read as template chrome; mono is reserved for identifiers and metrics) |
| Body-lg | Geist | 1.125rem | 400 | 1.6 | 0 |
| Body | Geist | 1rem | 400 | 1.6 | 0 |
| Small / meta | Geist | 0.875rem | 400 | 1.5 | 0 |
| Label | Geist | 0.8125rem | 500 | 1.4 | +0.01em |
| Mono / ID | Geist Mono | 0.875rem | 400 | 1.5 | 0; `tabular-nums` |

Recall IDs, UPCs, model and lot codes, metric values and score values always use Geist Mono with tabular numerals. Long-form measure is ≤ 68ch.

## 4. Space, shape, elevation, motion

- **Spacing:** 4px base. The scale is 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 (× 4px). Section rhythm is 96px desktop and 64px mobile.
- **Containers:**
  - `--container`: 1200px
  - `--container-wide`: 1360px (hero, search app)
  - `--measure`: 68ch
  - Gutters: 16px (< 640), 24px (< 1024), 40px.
- **Breakpoints (Tailwind defaults):**
  - sm 640
  - md 768 (filters inline)
  - lg 1024 (search split view)
  - xl 1280
  - Verified at 375, 768, 1024 and 1440.
- **Radius:**
  - `--radius-sm` 6px (badges, chips)
  - `--radius-md` 10px (inputs, buttons)
  - `--radius-lg` 16px (panels)
  - `--radius-pill` 999px (search bar, segmented control)
  - Not every container is a rounded card. Sections are separated by hairlines and whitespace, and lists are rows, not cards.
- **Elevation:** borders first. One shadow token, `--shadow-raised`, is used only for menus, the sheet and the hero search bar. Glow is limited to the hero visual and the primary-button hover (a 1px gold inner light), never on cards.
- **Motion:**
  - `--ease-out: cubic-bezier(.2,.7,.2,1)`
  - Durations 120ms (hover), 200ms (state), 320ms (panel/sheet), 600ms (entrance)
  - Hero orbit loop: 40–60s
  - Everything under `prefers-reduced-motion: reduce` becomes static or an instant fade.
  - There is no scroll-jacking and no WebGL. The hero visual is SVG + CSS (the cheapest technique that achieves the effect; the web-3d-motion skill was not needed and was not run), so a no-WebGL fallback is not needed.
- **Z-index:** base 0, sticky header 30, sheet/overlay 40, toast 50.
- **Touch targets:** ≥ 44 × 44px on all interactive elements below `md`.

## 5. API contract (from `backend/app/schemas.py`, not assumed)

| Endpoint | Used for | Key fields |
|---|---|---|
| `GET /search/models` | model picker and Model Lab availability | `models[]{model, available, score_type, description, reason}` |
| `POST /search` `{query (1–500), model, top_k (1–50)}` | results | `score_type`, `score_description`, `results[]{rank, recall_id, title, category, issue, url, score}`, `identifiers_in_query`, `identifier_checks[]{recall_id, verdict, by_type{upc,model,lot,best_before}}`, `notice` |
| `POST /explain` `{query, model, top_k (1–5)}` | evidence summary | `evidence_status`, `retrieved_facts[]`, `skipped_identifier_mismatches`, `explanation{generator, text, cited_recall_ids, fallback_reason}`, `sources[]`, `disclaimer` |
| `GET /recalls/{id}` | detail | `RecallDetail`: title, url, domain, organization, category, categories[], recall_class, last_updated, archived, product, issue, what_you_should_do, product_family, … |
| `GET /categories` | filters | `items[]{category, domain, count}` |

- Model keys are `bm25`, `minilm`, `minilm_finetuned` and `minilm_hard_negative`. The UI labels are BM25, Model A, Model B and Model C.
- `NEXT_PUBLIC_API_BASE_URL` is the only frontend configuration (already in `.env.example`). The backend's `cors_origins` already defaults to `http://localhost:3000`.
- No checkpoints, keys or server paths reach the browser. A 503 `detail` from an unavailable model is shown as "Model unavailable on this server", without echoing server paths.

**Integration gap — needs a decision:**
- `POST /search` has **no category filter**. The fields search hits lack (`recall_class`, `last_updated`) come from `GET /recalls/{id}` for the selected record only.
- See the open decision in §11.

## 6. Page architecture

All routes share `SiteHeader` (sticky, translucent `--bg` at 80% with backdrop blur and a hairline bottom border) and `SiteFooter`.

### `/` — Landing (Server Component and two small client islands)

1. **Hero (two columns ≥ lg, stacked below)**
   - Left column:
     - Eyebrow "Canadian recall search · 1,000 official notices".
     - H1 (working copy, to be approved): "Find the recall notice that matches what's in your hands."
     - A one-sentence description.
     - `HeroSearch`: a pill input, an Enter/submit to `/search?q=…`, and three example-query chips.
       - The chips come from `src/data/examples.json`, generated by script from real indexed records, so the build never needs a running API.
     - A secondary link: "How it works".
   - Right column: `LabelScanVisual`. It is an original champagne-gold line drawing that shows the product's actual job:
     - A stylised product label (brand line, barcode, `UPC`, `LOT`, `BEST BEFORE` in mono) on a dark plate.
     - A gold scanning line passes over it. As it crosses a code, that code lights up and a thin connector draws to a small notice excerpt card with a MATCH badge.
     - The orbital and scanning geometry from the brand direction lives in the faint concentric guides behind the label, with a soft radial light.
     - The codes are illustrative placeholders and are labelled as an example in an `aria-hidden` visual. There is a static frame under reduced motion.
2. **How it works**: three steps in an asymmetric row, not identical cards. Each step has a small mono artifact, e.g. a sample verdict line.
   - Describe (plain words or codes).
   - Find the closest notices (four retrieval models; keyword and semantic).
   - Verify (exact UPC, model, lot and date comparison, then read the official notice).
   - The capabilities (semantic search, identifier verification, evidence-only summaries, model transparency) are folded into these steps as one-line notes. There is no separate generic "Capabilities" grid.
3. **Official source and limits** (an editorial block with a hairline frame)
   - Data: Government of Canada Recalls and Safety Alerts (recalls-rappels.canada.ca).
   - The index is a 1,000-notice sample.
   - A close notice is not confirmation.
   - No result never means safe.
   - Always check the official notice.
   - English only: official notices are also available in French.
4. **Model Lab entry**: one line on the four approaches with BM25 as the baseline, linking to `/model-lab`. There is no single cherry-picked metric on the consumer page.
5. **Footer**
   - Nav, data attribution, course context (ARTE 4110U · Group 5), the disclaimer and the theme toggle.
   - No social icons, no newsletter.

### `/search` — Search app (client component; all state in the URL)

URL state is `?q=…&model=…&domain=…&cat=…&id=…`. Shared links restore everything.
- Changes to `q`, `model` and the filters **push** a history entry.
- Selecting a row (`id`) uses **`router.replace`**, so one Back press returns to the previous query.
- The page is wrapped in `<Suspense>`, as Next 16 requires for `useSearchParams` under `cacheComponents`.
- The **default model is Model C** (`minilm_hard_negative`, the best proxy point estimate). If it is unavailable, the default falls back to the first available model in the order C → B → A → BM25, and the fallback is stated in the status line.

**Layout:**
- **Search row:**
  - `SearchForm` (`role="search"`): labelled input, submit and clear. This is all a consumer needs to touch.
- **Refine row (secondary):**
  - "Retrieval model" select, with the label and one-line description from `lib/models.ts`. Unavailable models are disabled and show "unavailable on this server".
  - Domain (Food / Consumer products).
  - Category select (counts from `GET /categories`).
  - On desktop these are inline native `<select>`s.
  - On mobile a single "Refine" button opens a `Sheet` containing the same three controls (this keeps the accessible filter sheet from the original spec).
- **Status line (always visible, including at 375px without scrolling):**
  - "10 closest notices · ranked by keyword overlap" (BM25), or "… · ranked by meaning (Model C)" for the semantic models.
  - When a semantic model is active: "Semantic models always return their nearest notices, even when none is about your product."
  - The coverage line from trust rule 5.
  - A persistent "Search all official recalls ↗" link (recalls-rappels.canada.ca).
  - The API `notice` text, in a quiet info callout.
- **Evidence summary:**
  - A one-line trigger directly under the status line, the same on desktop and mobile: "Summarise the evidence (top 5)".
  - It expands in place and calls `/explain` lazily. The result is **cleared whenever `q`, `model` or the filters change**, so a stale summary is never shown.
  - It renders:
    - The header "Generated summary (template) — not an official statement", with `generator` shown.
    - `explanation.text`.
    - `cited_recall_ids` as links that select those rows.
    - "n higher-ranked notices set aside because their codes differ from yours", when `skipped_identifier_mismatches > 0`.
    - `fallback_reason`, if present, in muted text.
    - The API `disclaimer`, verbatim and always visible with the text.
    - The `evidence_status` wording for `no_results` and `identifier_mismatch`.
- **Desktop ≥ lg (split view):**
  - The left 5/12 is the `ResultList`.
  - The right 7/12 is a sticky `ResultDetailPanel` for the selected id (`GET /recalls/{id}`). It shows:
    - An H2 title.
    - The official-notice button (primary).
    - The facts grid.
    - The identifier comparison (below).
    - "What you should do".
    - "Open full page".
  - Focus is **not** moved into the panel on selection. The H2 lets screen-reader users jump to it, and `scroll-padding-top` clears the sticky header.
- **Mobile < lg:**
  - Single column.
  - Each row with an identifier verdict shows its per-type comparison inline (expandable), so mobile users see the same evidence as desktop users before leaving the list.
  - Tapping a row navigates to `/recalls/[id]?from=<encoded q/model>`. "← Back to results" restores the query.
- **Result row:**
  - A real `<a href>` (works with middle-click and open-in-new-tab) with `aria-current` when selected.
  - Contents:
    - Rank (mono).
    - Title (2-line clamp).
    - Issue.
    - **Last updated** date (mono, labelled exactly).
    - An "Archived" marker when applicable.
    - A domain/category chip.
    - A verdict badge when `identifiers_in_query`.
  - No score or percentage appears in the row.
  - The raw score and `score_type` appear only in a "Technical details" `<details>` element in the detail panel. (The row date and archived fields depend on decision 11.1.)
- **Identifier comparison** (`IdentifierChecks`):
  - One mono line for each `by_type` entry where either list is non-empty: "UPC — You entered: 0 12345 67890 2 · Notice lists: 0 12345 67890 2 · Matched: 0 12345 67890 2".
  - The verdict copy:
    - MATCH (`--alert`, strongest): "Your code appears in this notice — read the notice and follow its instructions."
    - MISMATCH (`--neutral`): "Your code isn't listed in this notice — check the notice for all affected codes."
    - UNKNOWN (`--unknown`): "Codes could not be compared (missing from your search or from this record)."
- **States:**
  - **Idle:** example queries (static JSON) and the hint "Try a brand, product, or a UPC/lot code". An optional "Recently updated notices" list from `GET /recalls?limit=5`.
  - **Loading:** skeleton rows and `aria-busy`. The previous results are dimmed rather than removed.
  - **Empty (BM25 only in practice):** "No notices in the RecallIQ sample matched these words." plus the official search link. The page never says "safe".
  - **Model unavailable (503):** inline, with a switch-model action. The server `detail` is never echoed.
  - **Validation (422):** a field message tied with `aria-describedby`.
  - **Network down:** "Can't reach the RecallIQ API" with a retry.
- **Announcements:** an `aria-live="polite"` region reports result counts ("10 closest notices for …"), model changes and errors.

### `/recalls/[id]` — Recall detail (Server Component, `GET /recalls/{id}`)

- Header:
  - Eyebrow (domain · organization).
  - H1 title.
  - Meta in mono: recall ID, "Last updated", category, recall class, archived flag.
  - **"Read the official notice ↗"** as the primary action.
- The coverage line (trust rule 5) sits under the header.
- Sections, shown only when a field is present:
  - Product
  - Issue
  - What you should do
  - Categories
- `product_family` is shown only as "RecallIQ grouping (not from the notice)". When `title_group_size > 1`, a "Related notices (n)" link points to a search for that group.
- Absent fields are omitted, or show "Not listed in this record" where their absence matters (e.g. product).
- **Identifier panel:** `RecallDetail` carries no identifier data. When `from` holds a query, a small client island re-runs `POST /search` with that `q` and `model` (`top_k` 50) and shows this `recall_id`'s `identifier_checks` entry with the same `IdentifierChecks` component. If the record is not in that result set, it says so. Without `from`: "Add a UPC, model or lot code to your search to compare identifiers."
- 404 renders a designed `not-found` view. API errors render `error.tsx` with a retry.
- Internal review fields (`needs_review`, `family_confidence`, …) are **not** shown to consumers.

### `/model-lab` — Four-model comparison (Server Component, static data)

- The data is `src/data/model-lab.json`, generated by `scripts/export-model-lab.mjs` from `experiments/runs/proxy_validation.json` and the run records. The script fails the build if a field is missing; no number is typed by hand.
- **Layout:**
  - **Header:** the evaluation setup in one paragraph, plus a prominent **PROXY** label (info colour + text): "Generated validation queries (391), not human-written. Independent human evaluation pending; the test split has not been evaluated."
  - **Model rows:** BM25, A, B and C. Each row has the name, the method in one line, the key settings in mono, and live availability from `GET /search/models`.
  - **Metrics table:** R@5, MRR@10, nDCG@10 and FMR@1 (with counts), all mono. The best value in each column is marked with a text marker, not only colour.
  - **Comparison chart:** a dot plot of paired-bootstrap differences with 95% CIs (B−A, C−B, B−BM25, C−BM25), drawn on a zero line in SVG with a table alternative. This is the honest way to show that C−B overlaps zero.
  - **Try a query: compare models.** One input runs 4 parallel `POST /search` calls and shows the top-5 ranked titles in four columns (stacked on mobile).
    - It shows ranks only, never cross-model scores (trust rule 2).
    - Rows that differ between models are marked with a text marker.
    - This is the assessor's live side-by-side view.
  - **Training curves:** the existing SVGs from `experiments/reports/`, re-rendered with theme tokens, or embedded as images with captions.
  - **Limitations:**
    - Proxy saturation (BM25 MRR ≥ 0.95).
    - Single seed for v1, plus a multi-seed note once those runs finish.
    - Identifier precision is not computable (6 of 1,000 records).
    - Label noise in the generated queries.

### `/how-it-works` — Method, data, limits (Server Component, static)

1. The pipeline diagram (SVG): query → retrieval (4 models) → identifier verification → evidence selection → template explanation → official source.
2. The data source and snapshot facts (1,000 records, food and consumer products, frozen split).
3. What the identifier verdicts mean, with real examples of the badges.
4. What the explanation is and is not (a deterministic template, no LLM in use, a grounding guard).
5. Limitations and responsible use.
6. A short FAQ: "Does no result mean my product is safe?" → No.

### Global

- `not-found.tsx`, `error.tsx` and a skip-to-content link.
- Per-route `metadata` (title and description).
- One `<h1>` per page.

## 7. Component plan

```
src/
  app/
    layout.tsx               fonts, theme script, header/footer, skip link
    page.tsx                 landing
    search/page.tsx          Suspense → <SearchApp/>
    recalls/[id]/page.tsx    detail (+ not-found.tsx, error.tsx)
    model-lab/page.tsx
    how-it-works/page.tsx
    not-found.tsx  error.tsx  globals.css (tokens)
  lib/
    api/types.ts             TS mirror of backend schemas (hand-written, field-for-field)
    api/client.ts            fetch wrapper: base URL, timeout, typed ApiError(status, kind)
    models.ts                model key → label/short description
    search-params.ts         parse/serialize URL state
  data/model-lab.json        generated (do not edit)
  components/
    brand/      Wordmark (TEMPORARY), LabelScanVisual
    layout/     SiteHeader, MobileNav, SiteFooter, ThemeToggle, Container, Section, Eyebrow
    ui/         Button (primary|secondary|ghost; sm|md), IconButton, TextField/SearchField,
                SegmentedControl, Select, Chip (filter, toggle), Badge, VerdictBadge,
                Callout (info|warning|danger), Skeleton, EmptyState, ErrorState, Sheet,
                ExternalLink (↗ + "opens official site"), Icon (inline SVG set, ~12 icons)
    search/     SearchApp, SearchForm, ModelPicker, CategoryFilter, ResultList, ResultRow,
                ResultDetailPanel, IdentifierChecks, ExplanationPanel, StatusLine
    recall/     RecallHeader, RecallFacts, RecallSection
    lab/        ModelRow, MetricsTable, DifferenceDotPlot, ProxyBadge, LimitationsList
scripts/export-model-lab.mjs
```

**Key component specs:**
- **`VerdictBadge`:**
  - MATCH: alert icon + "Code in notice", `--alert`, filled tint (strongest).
  - MISMATCH: cross icon + "Code not listed", `--neutral`, outline only.
  - UNKNOWN: dash icon + "Not comparable", `--unknown`, outline only.
  - It has an accessible name. The full comparison lives in `IdentifierChecks`.
- **`ThemeToggle`:** a `radiogroup` (System / Dark / Light).
- **`Button`:**
  - Primary: gold fill with obsidian text.
  - Secondary: hairline with `--text`.
  - Ghost.
  - States: hover, active, focus-visible ring, disabled (40% opacity + `aria-disabled`), loading (spinner + label kept).
- **`SegmentedControl`:** a `radiogroup` with arrow-key navigation; it collapses to `Select` below md.
- **`Sheet`:** `role="dialog"`, `aria-modal`, a focus trap, return focus to the trigger, and Esc to close.
- **`ExternalLink`:** always `rel="noopener noreferrer"`, `target="_blank"`, with visually hidden text "(opens in a new tab)".

**Icons:** a small inline-SVG set (search, arrow, external, check, cross, dash, filter, close, menu, sun, moon, info), drawn as 1.5px strokes. This avoids adding an icon dependency.

## 8. Motion plan

| Where | What | Reduced motion |
|---|---|---|
| Hero visual | scan line 7s sweep over the label; codes light and the connector draws as it passes; background orbit guides rotate 60s | static frame, MATCH state shown |
| Hero copy | 600ms stagger fade/translate-y 8px on first load | none |
| Search field | border to gold + 1px inner light on focus (200ms) | colour only |
| Result selection | 200ms background tint; detail panel content cross-fade 200ms | instant |
| Sheet | 320ms slide-up | instant |
| Section entrance | none (avoid scroll-triggered effects on content pages) | — |

These use CSS keyframes and transitions only; no animation library.

## 9. Accessibility requirements

These are design requirements; verification happens in Stage 7.
- Landmarks: header, nav, main and footer. Headings in order. A skip link.
- Every control is keyboard-operable, with a visible 2px focus ring and logical tab order.
- Forms: labelled inputs; errors are tied with `aria-describedby`; the search is a `role="search"` form.
- Live region for result counts and errors.
- Charts have a table alternative and a text summary.
- Touch targets ≥ 44px; no horizontal overflow at 320px; text zoom to 200% keeps the layout usable.
- Contrast is per §2.3; status is never colour-only.

## 10. Verification plan (Stage 8)

- `npm run lint`, `npx tsc --noEmit` and `npm run build`.
- The live FastAPI on :8000 is exercised for:
  - all 4 models
  - an identifier query (MATCH, MISMATCH and UNKNOWN)
  - the empty query
  - 404 recall
  - API down
- Screenshots at 375px and 1440px for every route (dark and light), via the Playwright MCP.
- A keyboard-only pass on `/search`.
- A Lighthouse accessibility audit via Chrome DevTools MCP.
- A `ui-ux-audit` review and an independent `product-designer` review.

## 11. Open decisions (need approval)

1. **Search filters and row metadata.** Choose one:
   - **(a)** One small, additive backend change with tests:
     - optional `domain` and `category` filters on `SearchRequest`, applied *before* top-k in `SearchEngine`;
     - `last_updated`, `archived` and `domain` added to `SearchHit`.
     - The filters become truthful, and rows can show the date and archived status.
   - **(b)** No backend change: client-side "filter these results" over a top-50 fetch, clearly labelled, and rows stay undated.
   - Recommendation: (a). Without dates, a 2014 archived notice looks like a current alert.
2. **Logo.** Supply the selected logo file(s), or approve the temporary wordmark for the checkpoint.
3. **Hero headline copy** in §6 is working copy. Approve it or replace it.
4. **Hero visual.** The spec uses the label-scan concept, with orbital guides kept as background. The alternative is the pure champagne orbital/scanning abstraction from the original brief.
5. **Default search model.** The spec uses Model C. The alternative is BM25, which is statistically indistinguishable on the proxy and the most explainable.

## 13. Review log

- **2026-11-16, independent `product-designer` agent review of v2** (15 findings, document-only).
- **Adopted:**
  - Verdict colours remapped (no green MATCH; MISMATCH is calm; errors-only red).
  - "Closest notices" framing.
  - Persistent coverage line and official search link.
  - Identifier evidence on the mobile/detail path (detail page re-runs the search).
  - The full `by_type` comparison shown.
  - Model picker moved to a secondary control with a documented default.
  - Row date and archived status (via decision 11.1).
  - `product_family` labelled as derived.
  - Explanation rendering and staleness rules.
  - History via `replace` for selection; rows as real links.
  - Landing: domain-specific hero, no generic capabilities grid, no single cherry-picked metric.
  - Side-by-side model comparison.
  - Static example data (the build works offline).
  - Two-tone focus ring, live-region scope, the English-only note.
  - Wording fixes.
- **Kept against the review:** the mobile filter `Sheet`. With model, domain and category it holds three controls, and the original spec asked for "Filters in an accessible sheet".

## 12. Out of scope (this phase)

Docker, hosting, accounts and sign-in, pricing, testimonials, customer logos, LLM provider integration, ML or backend changes beyond decision 11.1, and evaluation of the test split.
