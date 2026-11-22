"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { api, errorMessage } from "@/lib/api/client";
import { readSearchState, searchHref } from "@/lib/search-params";
import { IdentifierChecks } from "../search/IdentifierChecks";
import { useApi } from "../search/use-api";
import { Callout } from "../ui/Callout";
import { Icon } from "../ui/Icon";

/** "Back to results" for a recall page reached from a search; keeps the exact query and filters. */
export function BackToResults() {
  const state = readSearchState(useSearchParams());
  return (
    <Link href={state.q ? searchHref(state) : "/search"} className="inline-flex min-h-11 items-center gap-2 text-sm text-fg-2 hover:text-fg">
      <Icon name="arrowLeft" size={16} />
      {state.q ? "Back to results" : "Search recall notices"}
    </Link>
  );
}

/**
 * RecallDetail carries no identifier data, so when this page was opened from a search the
 * same search is re-run (top 50) and this notice's identifier comparison is shown.
 */
export function RecallIdentifierPanel({ recallId }: { recallId: number }) {
  const state = readSearchState(useSearchParams());
  const request = state.q
    ? { query: state.q, model: state.model, top_k: 50, domain: state.domain, category: state.cat }
    : null;
  const { data, error, loading, retry } = useApi(request ? JSON.stringify(request) : null, (signal) => api.search(request!, signal));

  let body;
  if (!request) {
    body = <p className="text-fg-2">Search with a UPC, model or lot code from the package to compare it with this notice.</p>;
  } else if (error) {
    body = (
      <Callout tone="danger" role="alert" title="Couldn't compare codes" action={<button type="button" className="btn btn-secondary" onClick={retry}>Try again</button>}>
        {errorMessage(error)}
      </Callout>
    );
  } else if (loading || !data) {
    body = <div className="skeleton h-16" aria-busy="true" aria-label="Comparing codes" />;
  } else if (!data.identifiers_in_query) {
    body = (
      <p className="text-fg-2">
        Your search (<span className="text-fg">{state.q}</span>) had no UPC, model or lot code to compare. Add one from the
        package to check it against this notice.
      </p>
    );
  } else {
    const check = data.identifier_checks.find((c) => c.recall_id === recallId);
    body = check ? (
      <IdentifierChecks check={check} />
    ) : (
      <p className="text-fg-2">This notice is not among the top 50 notices for your search, so its codes were not compared.</p>
    );
  }

  return (
    <section aria-labelledby="codes" className="rounded-[16px] border border-line bg-surface p-5 sm:p-6">
      <h2 id="codes" className="h3">
        Your codes compared with this notice
      </h2>
      <div className="mt-4">{body}</div>
    </section>
  );
}
