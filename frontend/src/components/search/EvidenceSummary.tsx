"use client";

import { useState } from "react";
import { api, errorMessage } from "@/lib/api/client";
import type { SearchRequest } from "@/lib/api/types";
import { Callout } from "../ui/Callout";
import { ExternalLink } from "../ui/ExternalLink";
import { useApi } from "./use-api";

/**
 * Lazy POST /explain over the top 5. Mounted with a key per search, so a summary for an
 * earlier query or model is never left on screen.
 */
export function EvidenceSummary({ request, onCite }: { request: SearchRequest; onCite: (id: number) => void }) {
  const [open, setOpen] = useState(false);
  const { data, error, loading, retry } = useApi(open ? JSON.stringify(request) : null, (signal) =>
    api.explain({ ...request, top_k: 5 }, signal),
  );

  return (
    <div className="rounded-[10px] border border-line">
      <button
        type="button"
        aria-expanded={open}
        aria-controls="evidence-summary"
        onClick={() => setOpen((value) => !value)}
        className="flex min-h-12 w-full items-center justify-between gap-4 px-4 text-left text-[0.9375rem] hover:text-fg"
      >
        <span>
          <span className="font-medium text-fg">Summarise the evidence</span>
          <span className="text-fg-3"> from the top 5 notices</span>
        </span>
        <span aria-hidden className={"text-fg-3 transition-transform duration-200 " + (open ? "rotate-45" : "")}>
          +
        </span>
      </button>
      <div id="evidence-summary" hidden={!open} className="border-t border-line px-4 py-4">
        {error ? (
          <Callout tone="danger" role="alert" title="Couldn't build the summary" action={<button type="button" onClick={retry} className="btn btn-secondary">Try again</button>}>
            {errorMessage(error)}
          </Callout>
        ) : loading || !data ? (
          <div aria-busy="true" aria-label="Building the summary" className="space-y-2">
            <div className="skeleton h-4 w-3/4" />
            <div className="skeleton h-4" />
            <div className="skeleton h-4 w-5/6" />
          </div>
        ) : (
          <div className="space-y-4 text-[0.9375rem]">
            <p className="text-[0.8125rem] text-fg-3">
              Generated summary ({data.explanation.generator === "template" ? "fixed template, no AI writing" : data.explanation.generator}), not an
              official statement.
            </p>
            {data.skipped_identifier_mismatches > 0 && (
              <p className="text-fg-2">
                {data.skipped_identifier_mismatches} higher-ranked{" "}
                {data.skipped_identifier_mismatches === 1 ? "notice was" : "notices were"} set aside because
                their codes differ from yours.
              </p>
            )}
            <p className="measure leading-relaxed text-fg">{data.explanation.text}</p>
            {data.retrieved_facts.length > 0 && (
              <div>
                <p className="text-sm text-fg-3">Based on</p>
                <ul className="mt-1.5 space-y-1 text-sm">
                  {data.retrieved_facts.map((fact) => (
                    <li key={fact.recall_id} className="flex flex-wrap items-baseline gap-x-2">
                      <button type="button" onClick={() => onCite(fact.recall_id)} className="link text-left">
                        {fact.title}
                      </button>
                      <ExternalLink href={fact.url} className="text-fg-3 hover:text-fg">
                        <span className="sr-only">Official notice for {fact.title}</span>
                      </ExternalLink>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {data.explanation.fallback_reason && <p className="text-sm text-fg-3">{data.explanation.fallback_reason}</p>}
            <p className="border-t border-line pt-3 text-sm text-fg-2">{data.disclaimer}</p>
          </div>
        )}
      </div>
    </div>
  );
}
