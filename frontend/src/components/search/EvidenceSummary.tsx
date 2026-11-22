"use client";

import { useState } from "react";
import { api, errorMessage } from "@/lib/api/client";
import type { EvidenceFact, ExplainResponse, SearchRequest } from "@/lib/api/types";
import { Callout } from "../ui/Callout";
import { ExternalLink } from "../ui/ExternalLink";
import { VerdictBadge } from "../ui/VerdictBadge";
import { useApi } from "./use-api";

/**
 * Lazy POST /explain (on open, so an AI answer is only paid for when asked). Mounted with a
 * key per search, so a summary for an earlier query or model is never left on screen.
 * top_k is left to the server (RECALLIQ_RAG_TOP_K).
 */
export function EvidenceSummary({ request, onCite }: { request: SearchRequest; onCite: (id: number) => void }) {
  const [open, setOpen] = useState(false);
  const { data, error, loading, retry } = useApi(open ? JSON.stringify(request) : null, (signal) =>
    api.explain({ ...request, top_k: undefined }, signal),
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
          <span className="text-fg-3"> from the top notices</span>
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
          <div aria-busy="true" className="space-y-3">
            <p role="status" className="text-sm text-fg-3">Reading the official notices and writing a summary…</p>
            <div className="space-y-2" aria-hidden>
              <div className="skeleton h-4 w-3/4" />
              <div className="skeleton h-4" />
              <div className="skeleton h-4 w-5/6" />
            </div>
          </div>
        ) : (
          <Summary data={data} onCite={onCite} />
        )}
      </div>
    </div>
  );
}

function Summary({ data, onCite }: { data: ExplainResponse; onCite: (id: number) => void }) {
  const exp = data.explanation;
  const ai = exp.generator.startsWith("llm:");
  const cited = new Set(exp.cited_recall_ids);
  const stats = exp.generation;
  return (
    <div className="space-y-4 text-[0.9375rem]">
      <p className="text-[0.8125rem] text-fg-3">
        {ai
          ? `AI-written summary (${stats?.model ?? "language model"}), checked against the official notices listed below; not an official statement.`
          : "Generated summary (fixed template, no AI writing), not an official statement."}
      </p>
      {exp.fallback_reason && (
        <Callout tone="quiet" role="status">
          {exp.fallback_reason}
        </Callout>
      )}
      {ai && exp.evidence_sufficient === false && (
        <Callout tone="info" title="These notices may not be about your product">
          Search the official database at{" "}
          <ExternalLink href="https://recalls-rappels.canada.ca/en">recalls-rappels.canada.ca</ExternalLink> as well.
        </Callout>
      )}
      {data.skipped_identifier_mismatches > 0 && (
        <p className="text-fg-2">
          {data.skipped_identifier_mismatches} higher-ranked{" "}
          {data.skipped_identifier_mismatches === 1 ? "notice was" : "notices were"} set aside because
          their codes differ from yours.
        </p>
      )}
      <p className="measure whitespace-pre-line leading-relaxed text-fg">{exp.text}</p>
      {data.retrieved_facts.length > 0 && (
        <div>
          <p className="text-sm text-fg-3">Sources: official Government of Canada notices</p>
          <ol className="mt-2 space-y-3">
            {data.retrieved_facts.map((fact) => (
              <Source key={fact.recall_id} fact={fact} cited={cited.has(fact.recall_id)} onCite={onCite} />
            ))}
          </ol>
        </div>
      )}
      <p className="border-t border-line pt-3 text-sm text-fg-2">{data.disclaimer}</p>
      {stats && stats.latency_ms !== null && (
        <p className="text-xs text-fg-3">
          {stats.model} · {(stats.latency_ms / 1000).toFixed(1)} s
          {stats.total_tokens !== null && ` · ${stats.total_tokens.toLocaleString()} tokens`}
        </p>
      )}
    </div>
  );
}

function Source({ fact, cited, onCite }: { fact: EvidenceFact; cited: boolean; onCite: (id: number) => void }) {
  const rows: [string, string | null][] = [
    ["Hazard", fact.issue],
    ["Product", fact.product],
    ["Recall class", fact.recall_class],
  ];
  return (
    <li className="rounded-[8px] border border-line px-3 py-2.5 text-sm">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <button type="button" onClick={() => onCite(fact.recall_id)} className="link text-left">
          {fact.title}
        </button>
        <ExternalLink href={fact.url} className="text-fg-3 hover:text-fg">
          <span className="sr-only">Official notice for {fact.title}</span>
        </ExternalLink>
      </div>
      <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[0.8125rem] text-fg-3">
        <span>Recall {fact.recall_id}</span>
        {cited && <span className="text-accent">· Cited in the summary</span>}
        {fact.identifier_verdict !== "UNKNOWN" && <VerdictBadge verdict={fact.identifier_verdict} />}
      </p>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[0.8125rem]">
        {rows.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-fg-3">{label}</dt>
            <dd className={value ? "text-fg-2" : "italic text-fg-3"}>{value ?? "Not listed"}</dd>
          </div>
        ))}
      </dl>
      {fact.what_you_should_do && (
        <details className="mt-2 text-[0.8125rem]">
          <summary className="cursor-pointer text-fg-2 hover:text-fg">What the notice says to do</summary>
          <p className="measure mt-1 whitespace-pre-line text-fg-2">{fact.what_you_should_do}</p>
        </details>
      )}
    </li>
  );
}
