"use client";

import Link from "next/link";
import { useState } from "react";
import { api, errorMessage } from "@/lib/api/client";
import type { SearchHit, SearchModel } from "@/lib/api/types";
import { MODELS } from "@/lib/models";
import { recallHref } from "@/lib/search-params";
import { EXAMPLE_QUERIES } from "@/data/examples";
import { useApi } from "../search/use-api";

type Outcome = { model: SearchModel; hits?: SearchHit[]; error?: string };

/** Same query, four models, top 5 side by side. Ranks only: scores are not comparable across models. */
export function CompareModels() {
  const [draft, setDraft] = useState<string>(EXAMPLE_QUERIES[0]);
  const [query, setQuery] = useState<string | null>(null);

  const { data, loading } = useApi(query, (signal) =>
    Promise.all(
      MODELS.map((m) =>
        api.search({ query: query!, model: m.key, top_k: 5 }, signal).then(
          (r): Outcome => ({ model: m.key, hits: r.results }),
          // One model failing (e.g. its checkpoint is missing: 503) must not hide the others,
          // so each request settles to an Outcome; only a cancellation propagates.
          (error): Outcome => {
            if (signal.aborted) throw error;
            return { model: m.key, error: errorMessage(error) };
          },
        ),
      ),
    ),
  );

  const topOf = new Map<number, number>(); // recall id -> how many models rank it first
  data?.forEach((o) => o.hits?.[0] && topOf.set(o.hits[0].recall_id, (topOf.get(o.hits[0].recall_id) ?? 0) + 1));

  return (
    <div>
      <form
        className="flex flex-col gap-2 sm:flex-row"
        onSubmit={(event) => {
          event.preventDefault();
          if (draft.trim()) setQuery(draft.trim());
        }}
      >
        <label htmlFor="compare-q" className="sr-only">
          Query to compare across models
        </label>
        <input id="compare-q" className="field min-h-12 flex-1" value={draft} maxLength={500} onChange={(e) => setDraft(e.target.value)} />
        <button type="submit" className="btn btn-primary min-h-12" disabled={!draft.trim()}>
          Compare the four models
        </button>
      </form>

      <div aria-live="polite" className="sr-only">
        {loading ? "Comparing models" : data ? "Comparison ready" : ""}
      </div>

      {query && (
        <div aria-busy={loading} className={"mt-6 grid gap-px overflow-hidden rounded-[16px] border border-line bg-line md:grid-cols-2 xl:grid-cols-4 " + (loading ? "opacity-60" : "")}>
          {MODELS.map((m) => {
            const outcome = data?.find((o) => o.model === m.key);
            return (
              <section key={m.key} aria-labelledby={"cmp-" + m.key} className="bg-canvas p-5">
                <h3 id={"cmp-" + m.key} className="font-medium">
                  {m.name} <span className="font-normal text-fg-3">{m.short}</span>
                </h3>
                {!outcome ? (
                  <div className="mt-4 space-y-3" aria-hidden>
                    {[0, 1, 2].map((i) => (
                      <div key={i} className="skeleton h-4" />
                    ))}
                  </div>
                ) : outcome.error ? (
                  <p className="mt-4 text-sm text-danger">{outcome.error}</p>
                ) : outcome.hits!.length === 0 ? (
                  <p className="mt-4 text-sm text-fg-3">No notice shares words with this query.</p>
                ) : (
                  <ol className="mt-3 space-y-2.5 text-sm">
                    {outcome.hits!.map((hit) => (
                      <li key={hit.recall_id} className="grid grid-cols-[1.25rem_1fr] gap-2">
                        <span className="num text-fg-3">{hit.rank}</span>
                        <Link href={recallHref(hit.recall_id)} prefetch={false} className="line-clamp-2 text-fg-2 hover:text-fg">
                          {hit.title}
                          {hit.rank === 1 && topOf.get(hit.recall_id) !== MODELS.length && (
                            <span className="ml-1.5 text-[0.75rem] text-accent">(top pick differs)</span>
                          )}
                        </Link>
                      </li>
                    ))}
                  </ol>
                )}
              </section>
            );
          })}
        </div>
      )}
    </div>
  );
}
