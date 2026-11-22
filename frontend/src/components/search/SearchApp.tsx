"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useMemo, useState, type MouseEvent } from "react";
import { EXAMPLE_QUERIES } from "@/data/examples";
import { api, ApiError, errorMessage } from "@/lib/api/client";
import type { SearchRequest } from "@/lib/api/types";
import { COVERAGE_NOTE, formatDate, OFFICIAL_SEARCH_URL } from "@/lib/format";
import { DEFAULT_MODEL, modelInfo } from "@/lib/models";
import { readSearchState, recallHref, searchHref, type SearchState } from "@/lib/search-params";
import { Callout } from "../ui/Callout";
import { ExternalLink } from "../ui/ExternalLink";
import { Icon } from "../ui/Icon";
import { Sheet } from "../ui/Sheet";
import { DetailPanel } from "./DetailPanel";
import { EvidenceSummary } from "./EvidenceSummary";
import { RefineControls, type RefineValue } from "./RefineControls";
import { ResultRow } from "./ResultRow";
import { useApi } from "./use-api";

const TOP_K = 10;
// Matches Tailwind's `lg` breakpoint: at this width results and the detail panel sit side
// by side, so selecting a row updates the panel instead of navigating to the recall page.
const WIDE = "(min-width: 1024px)";

/**
 * The /search page. The URL is the single source of truth (query, model, filters, selected
 * notice), so every state can be shared, bookmarked and restored with Back/Forward. Each
 * request is keyed by its JSON body: changing any input cancels the previous fetch in
 * `useApi`, so a slow response for an old query can never overwrite a newer one.
 */
export function SearchApp() {
  const params = useSearchParams();
  const router = useRouter();
  const state = readSearchState(params);
  const [sheetOpen, setSheetOpen] = useState(false);

  const models = useApi("models", (signal) => api.models(signal));
  const categories = useApi("categories", (signal) => api.categories(signal));

  const request: SearchRequest | null = useMemo(
    () => (state.q ? { query: state.q, model: state.model, top_k: TOP_K, domain: state.domain, category: state.cat } : null),
    [state.q, state.model, state.domain, state.cat],
  );
  const searchKey = request ? JSON.stringify(request) : null;
  const search = useApi(searchKey, (signal) => api.search(request!, signal));
  const recent = useApi(state.q ? null : "recent", (signal) => api.recent(5, signal));

  // While a new search loads, keep showing (dimmed) the previous results instead of blanking.
  const response = search.data ?? (search.loading ? search.previous : undefined);
  const results = response?.results ?? [];
  const checks = new Map(response?.identifier_checks.map((c) => [c.recall_id, c]));
  // A stale ?id= (not in the current results) falls back to the top result.
  const selected = results.find((h) => h.recall_id === state.id) ?? results[0];
  const status = models.data?.models.find((m) => m.model === state.model);
  const info = modelInfo(state.model);
  const dense = state.model !== "bm25";

  // q, model and filters push history; selecting a row replaces it (one Back = previous query).
  const go = (next: Partial<SearchState>) => router.push(searchHref({ ...state, id: null, ...next }));
  const select = (id: number) => router.replace(searchHref({ ...state, id }), { scroll: false });
  // Rows are real links to /recalls/[id] (work without JS, open in new tabs); only a left
  // click without Ctrl/Meta/Shift on a wide screen is intercepted to fill the side panel.
  const onRowClick = (event: MouseEvent<HTMLAnchorElement>, id: number) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
    if (window.matchMedia(WIDE).matches) {
      event.preventDefault();
      select(id);
    }
  };
  const refine = (value: RefineValue) => go(value);

  const filtersActive = Number(Boolean(state.domain)) + Number(Boolean(state.cat)) + Number(state.model !== DEFAULT_MODEL);
  const announce = search.loading
    ? "Searching"
    : search.error
      ? errorMessage(search.error)
      : response
        ? results.length + " closest notices for " + response.query + ", ranked by " + info.ranking
        : "";

  return (
    <div className="mx-auto w-full max-w-[1360px] px-4 pb-16 pt-6 sm:px-6 lg:px-10 lg:pt-10">
      <h1 className="sr-only">Search recall notices</h1>
      <p aria-live="polite" className="sr-only">
        {announce}
      </p>

      <SearchForm key={state.q} initial={state.q} onSubmit={(q) => go({ q })} />

      <div className="mt-4 hidden md:block">
        <RefineControls idPrefix="refine" value={state} onChange={refine} models={models.data?.models} categories={categories.data?.items} />
      </div>
      <div className="mt-3 flex items-center justify-between md:hidden">
        <p className="text-sm text-fg-3">
          Model: <span className="text-fg-2">{info.name}</span> ({state.model === "bm25" ? "keywords" : "meaning"})
          {state.domain || state.cat ? ", filtered" : ""}
        </p>
        <button type="button" className="btn btn-secondary" onClick={() => setSheetOpen(true)} aria-haspopup="dialog">
          <Icon name="filter" />
          Refine{filtersActive ? " (" + filtersActive + ")" : ""}
        </button>
      </div>
      <Sheet
        open={sheetOpen}
        onClose={() => setSheetOpen(false)}
        title="Refine search"
        footer={
          <button type="button" className="btn btn-primary w-full" onClick={() => setSheetOpen(false)}>
            Show results
          </button>
        }
      >
        <RefineControls idPrefix="sheet" stacked value={state} onChange={refine} models={models.data?.models} categories={categories.data?.items} />
        <p className="mt-5 text-sm text-fg-3">{info.method}</p>
      </Sheet>

      {status && !status.available && (
        <div className="mt-4">
          <Callout
            tone="danger"
            role="alert"
            title={info.name + " is unavailable on this server"}
            action={
              <button type="button" className="btn btn-secondary" onClick={() => go({ model: DEFAULT_MODEL })}>
                Switch to BM25
              </button>
            }
          >
            Choose another retrieval model to search.
          </Callout>
        </div>
      )}

      {!state.q ? (
        <IdleState recent={recent.data?.items} />
      ) : (
        <>
          <div className="mt-6 border-b border-line pb-4">
            <p className="text-[0.9375rem] text-fg">
              {search.loading && !response ? (
                "Searching..."
              ) : (
                <>
                  <span className="num">{results.length}</span> closest {results.length === 1 ? "notice" : "notices"}, ranked by {info.ranking}
                </>
              )}
            </p>
            {dense && (
              <p className="mt-1 text-sm text-fg-3">
                Semantic models always return their nearest notices, even when none is about your product.
              </p>
            )}
            <div className="mt-1 flex flex-wrap items-center gap-x-5 text-sm text-fg-3">
              <span className="py-2">{COVERAGE_NOTE}</span>
              <ExternalLink href={OFFICIAL_SEARCH_URL} className="link inline-flex min-h-11 items-center">
                Search all official recalls
              </ExternalLink>
            </div>
            {response?.notice && (
              <details className="text-sm">
                <summary className="inline-flex min-h-11 cursor-pointer items-center text-fg-2 hover:text-fg">How to read these results</summary>
                <p className="measure pb-1 text-fg-2">{response.notice}</p>
              </details>
            )}
          </div>

          {search.error ? (
            <div className="mt-6">
              <Callout
                tone="danger"
                role="alert"
                title={search.error instanceof ApiError && search.error.kind === "unavailable" ? info.name + " couldn't run" : "The search didn't complete"}
                action={
                  <div className="flex flex-wrap gap-2">
                    <button type="button" className="btn btn-secondary" onClick={search.retry}>
                      <Icon name="refresh" /> Try again
                    </button>
                    {state.model !== DEFAULT_MODEL && (
                      <button type="button" className="btn btn-ghost" onClick={() => go({ model: DEFAULT_MODEL })}>
                        Search with BM25
                      </button>
                    )}
                  </div>
                }
              >
                {errorMessage(search.error)}
              </Callout>
            </div>
          ) : response && results.length === 0 && !search.loading ? (
            <EmptyState filtered={Boolean(state.domain || state.cat)} onClear={() => go({ domain: null, cat: null })} />
          ) : (
            <div className="mt-6 grid gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:gap-10">
              <div className="min-w-0 space-y-4">
                {request && response && !search.loading && (
                  <EvidenceSummary key={searchKey} request={request} onCite={(id) => {
                    if (window.matchMedia(WIDE).matches) select(id);
                    else router.push(recallHref(id, state));
                  }} />
                )}
                <div aria-busy={search.loading} className={"overflow-hidden rounded-[16px] border border-line transition-opacity duration-200 " + (search.loading ? "opacity-50" : "")}>
                  {!response ? (
                    <ResultSkeleton />
                  ) : (
                    <ol aria-label="Closest notices">
                      {results.map((hit) => (
                        <ResultRow
                          key={hit.recall_id}
                          hit={hit}
                          href={recallHref(hit.recall_id, state)}
                          check={checks.get(hit.recall_id)}
                          showCheck={response.identifiers_in_query}
                          selected={hit.recall_id === selected?.recall_id}
                          onSelect={onRowClick}
                        />
                      ))}
                    </ol>
                  )}
                </div>
                {response && !response.identifiers_in_query && (
                  <p className="text-sm text-fg-3">
                    Add a UPC, model or lot code from the package to compare codes with each notice.
                  </p>
                )}
              </div>
              <div className="hidden lg:block">
                <div className="rounded-[16px] border border-line bg-surface p-7">
                  {response && selected ? (
                    <DetailPanel
                      key={selected.recall_id}
                      hit={selected}
                      check={checks.get(selected.recall_id)}
                      response={response}
                      fullHref={recallHref(selected.recall_id, state)}
                    />
                  ) : (
                    <div className="space-y-4" aria-hidden>
                      <div className="skeleton h-4 w-24" />
                      <div className="skeleton h-8 w-4/5" />
                      <div className="skeleton h-11 w-56" />
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function SearchForm({ initial, onSubmit }: { initial: string; onSubmit: (q: string) => void }) {
  const [value, setValue] = useState(initial);
  const [error, setError] = useState(false);
  return (
    <form
      role="search"
      aria-label="Search recall notices"
      noValidate
      onSubmit={(event) => {
        event.preventDefault();
        const q = value.trim();
        if (!q) return setError(true);
        onSubmit(q);
      }}
    >
      <label htmlFor="q" className="mb-2 block text-sm font-medium text-fg-2">
        Describe the product
      </label>
      <div className="flex gap-2">
        <div className="relative min-w-0 flex-1">
          <Icon name="search" className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-fg-3" />
          <input
            id="q"
            type="search"
            value={value}
            maxLength={500}
            autoComplete="off"
            autoFocus={!initial}
            placeholder="Brand, product, or a code from the package"
            aria-invalid={error || undefined}
            aria-describedby={error ? "q-error" : undefined}
            onChange={(event) => {
              setValue(event.target.value);
              if (error) setError(false);
            }}
            className="field min-h-12 pl-11 text-base"
          />
        </div>
        <button type="submit" className="btn btn-primary min-h-12 px-5 sm:px-7">
          Search
        </button>
      </div>
      {error && (
        <p id="q-error" role="alert" className="mt-2 text-sm text-danger">
          Describe a product first, such as its brand or what it is.
        </p>
      )}
    </form>
  );
}

function IdleState({ recent }: { recent?: { id: number; title: string; last_updated: string | null }[] }) {
  return (
    <div className="mt-10 grid gap-10 lg:grid-cols-2">
      <div>
        <h2 className="h3">Start with what&apos;s on the package</h2>
        <p className="mt-2 max-w-md text-fg-2">
          The brand and product name work well. Adding a UPC, model or lot code lets RecallIQ compare it exactly
          with each notice.
        </p>
        <ul className="mt-5 flex flex-wrap gap-2">
          {EXAMPLE_QUERIES.map((example) => (
            <li key={example}>
              <Link href={searchHref({ q: example })} className="inline-flex min-h-11 items-center rounded-full border border-line px-4 text-sm text-fg-2 hover:border-accent hover:text-fg">
                {example}
              </Link>
            </li>
          ))}
        </ul>
        <p className="mt-6 text-sm text-fg-3">{COVERAGE_NOTE}</p>
      </div>
      <div>
        <h2 className="h3">Recently updated in this sample</h2>
        {recent ? (
          <ul className="mt-4 divide-y divide-line border-y border-line">
            {recent.map((item) => (
              <li key={item.id}>
                <Link href={"/recalls/" + item.id} prefetch={false} className="flex flex-col gap-1 py-3.5 hover:text-accent sm:flex-row sm:items-baseline sm:justify-between sm:gap-6">
                  <span className="line-clamp-2">{item.title}</span>
                  <span className="tnum shrink-0 text-sm text-fg-3">{formatDate(item.last_updated) ?? "date not listed"}</span>
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <div className="mt-4 space-y-3" aria-hidden>
            {[0, 1, 2].map((i) => (
              <div key={i} className="skeleton h-6" />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function EmptyState({ filtered, onClear }: { filtered: boolean; onClear: () => void }) {
  return (
    <div className="mt-10 max-w-xl">
      <h2 className="h3">
        {filtered
          ? "No notices in this sample match your words and filters"
          : "No notices in this sample share words with your description"}
      </h2>
      <p className="mt-3 text-fg-2">
        That says nothing about whether the product is safe. Try other words, such as the brand or product type,
        {filtered ? " remove the filters," : ""} or search the full official database.
      </p>
      <div className="mt-6 flex flex-wrap gap-3">
        <ExternalLink href={OFFICIAL_SEARCH_URL} className="btn btn-primary">
          Search all official recalls
        </ExternalLink>
        {filtered && (
          <button type="button" className="btn btn-secondary" onClick={onClear}>
            Remove filters
          </button>
        )}
      </div>
    </div>
  );
}

function ResultSkeleton() {
  return (
    <div className="divide-y divide-line" aria-hidden>
      {[0, 1, 2, 3, 4].map((i) => (
        <div key={i} className="grid grid-cols-[2rem_1fr] gap-3 px-5 py-4">
          <div className="skeleton h-4 w-4" />
          <div className="space-y-2">
            <div className="skeleton h-4 w-4/5" />
            <div className="skeleton h-3 w-1/2" />
          </div>
        </div>
      ))}
    </div>
  );
}
