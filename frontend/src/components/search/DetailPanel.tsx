"use client";

import Link from "next/link";
import { api, errorMessage } from "@/lib/api/client";
import type { IdentifierCheck, SearchHit, SearchResponse } from "@/lib/api/types";
import { DOMAIN_LABEL, formatDate } from "@/lib/format";
import { Callout } from "../ui/Callout";
import { ExternalLink } from "../ui/ExternalLink";
import { IdentifierChecks } from "./IdentifierChecks";
import { useApi } from "./use-api";

function Fact({ term, value, mono = false }: { term: string; value: string | null; mono?: boolean }) {
  return (
    <div className="py-3">
      <dt className="text-[0.8125rem] text-fg-3">{term}</dt>
      <dd className={"mt-0.5 " + (value ? "text-fg" : "text-fg-3") + (mono && value ? " num" : "")}>
        {value ?? "Not listed in this record"}
      </dd>
    </div>
  );
}

/** Wide-screen side panel for the selected notice. Focus is not moved here on selection. */
export function DetailPanel({
  hit,
  check,
  response,
  fullHref,
}: {
  hit: SearchHit;
  check?: IdentifierCheck;
  response: SearchResponse;
  fullHref: string;
}) {
  const { data: recall, error, loading, retry } = useApi(String(hit.recall_id), (signal) => api.recall(hit.recall_id, signal));

  return (
    <section aria-labelledby="detail-title" className="fade-in" key={hit.recall_id}>
      <p className="text-sm text-fg-3">
        Rank <span className="num">{hit.rank}</span> of {response.results.length}
      </p>
      <h2 id="detail-title" className="h2 mt-2">
        {hit.title}
      </h2>
      <div className="mt-5 flex flex-wrap gap-3">
        <ExternalLink href={hit.url} className="btn btn-primary">
          Read the official notice
        </ExternalLink>
        <Link href={fullHref} prefetch={false} className="btn btn-secondary">
          Open full page
        </Link>
      </div>

      {response.identifiers_in_query && check && (
        <div className="mt-8">
          <h3 className="mb-3 font-medium">Your codes compared with this notice</h3>
          <IdentifierChecks check={check} />
        </div>
      )}

      <div className="mt-8">
        {error ? (
          <Callout tone="danger" role="alert" title="Couldn't load this notice's details" action={<button type="button" onClick={retry} className="btn btn-secondary">Try again</button>}>
            {errorMessage(error)}
          </Callout>
        ) : loading || !recall ? (
          <div className="space-y-3" aria-busy="true" aria-label="Loading notice details">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="skeleton h-10" />
            ))}
          </div>
        ) : (
          <>
            <div className="space-y-6">
              {(
                [
                  ["Issue", recall.issue],
                  ["What you should do", recall.what_you_should_do],
                  ["Product", recall.product],
                ] as const
              ).map(([title, text]) =>
                text ? (
                  <div key={title}>
                    <h3 className="font-medium">{title}</h3>
                    <p className="measure mt-1.5 whitespace-pre-line text-fg-2">{text}</p>
                  </div>
                ) : null,
              )}
            </div>
            <dl className="mt-8 grid grid-cols-1 divide-y divide-line border-y border-line sm:grid-cols-2 sm:divide-y-0 sm:[&>*]:border-b sm:[&>*]:border-line">
              <Fact term="Last updated" value={formatDate(recall.last_updated)} />
              <Fact term="Status" value={recall.archived ? "Archived notice" : "Active notice"} />
              <Fact term="Issued by" value={recall.organization} />
              <Fact term="Recall class" value={recall.recall_class} />
              <Fact term="Type" value={DOMAIN_LABEL[recall.domain]} />
              <Fact term="Categories" value={recall.categories.join(", ") || null} />
            </dl>
          </>
        )}
      </div>

      <details className="mt-8 border-t border-line pt-4 text-sm">
        <summary className="inline-flex min-h-11 cursor-pointer items-center text-fg-2 hover:text-fg">Technical details</summary>
        <dl className="mt-2 space-y-2 text-fg-2">
          <div>
            <dt className="text-fg-3">Recall ID</dt>
            <dd className="num">{hit.recall_id}</dd>
          </div>
          <div>
            <dt className="text-fg-3">Ranking score ({response.score_type === "bm25" ? "BM25" : "cosine similarity"})</dt>
            <dd className="num">{hit.score.toFixed(4)}</dd>
          </div>
          <div>
            <dt className="text-fg-3">How to read it</dt>
            <dd>{response.score_description} It is not a probability that your product is recalled.</dd>
          </div>
        </dl>
      </details>
    </section>
  );
}
