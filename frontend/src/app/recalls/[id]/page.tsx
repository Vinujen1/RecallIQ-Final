import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { Suspense } from "react";
import { BackToResults, RecallIdentifierPanel } from "@/components/recall/RecallIdentifierPanel";
import { Callout } from "@/components/ui/Callout";
import { ExternalLink } from "@/components/ui/ExternalLink";
import { api, ApiError } from "@/lib/api/client";
import { COVERAGE_NOTE, DOMAIN_LABEL, formatDate, OFFICIAL_SEARCH_URL } from "@/lib/format";

export const metadata: Metadata = { title: "Recall notice" };

const parseId = (raw: string) => {
  const id = Number(raw);
  return Number.isSafeInteger(id) && id > 0 ? id : null;
};

export default function RecallPage({ params }: { params: Promise<{ id: string }> }) {
  return (
    <div className="mx-auto w-full max-w-[1100px] px-4 pb-20 pt-4 sm:px-6 lg:px-10 lg:pt-8">
      <Suspense fallback={<span className="inline-flex min-h-11 items-center text-sm text-fg-3">Back</span>}>
        <BackToResults />
      </Suspense>
      <Suspense fallback={<RecallSkeleton />}>
        <Recall params={params} />
      </Suspense>
    </div>
  );
}

async function Recall({ params }: { params: Promise<{ id: string }> }) {
  const id = parseId((await params).id);
  if (id === null) notFound();
  let recall;
  try {
    recall = await api.recall(id);
  } catch (error) {
    if (error instanceof ApiError && error.kind === "not_found") notFound();
    throw error; // error.tsx offers a retry
  }

  const updated = formatDate(recall.last_updated);
  const sections = [
    ["Product", recall.product],
    ["Issue", recall.issue],
    ["What you should do", recall.what_you_should_do],
  ] as const;

  return (
    <article className="mt-4">
      <header className="border-b border-line pb-8">
        <p className="kicker">
          {DOMAIN_LABEL[recall.domain]} recall, issued by {recall.organization}
        </p>
        <h1 className="h1 mt-3 max-w-4xl">{recall.title}</h1>
        <dl className="mt-6 flex flex-wrap gap-x-8 gap-y-3 text-sm">
          <div>
            <dt className="text-fg-3">Last updated</dt>
            <dd className="tnum mt-0.5 text-fg">{updated ?? "Not listed"}</dd>
          </div>
          <div>
            <dt className="text-fg-3">Status</dt>
            <dd className="mt-0.5 text-fg">{recall.archived ? "Archived notice" : "Active notice"}</dd>
          </div>
          <div>
            <dt className="text-fg-3">Recall class</dt>
            <dd className="mt-0.5 text-fg">{recall.recall_class ?? "Not listed"}</dd>
          </div>
          <div>
            <dt className="text-fg-3">Recall ID</dt>
            <dd className="num mt-0.5 text-fg">{recall.id}</dd>
          </div>
        </dl>
        <div className="mt-7 flex flex-wrap items-center gap-3">
          <ExternalLink href={recall.url} className="btn btn-primary">
            Read the official notice
          </ExternalLink>
          <ExternalLink href={OFFICIAL_SEARCH_URL} className="btn btn-ghost">
            Search all official recalls
          </ExternalLink>
        </div>
        <p className="mt-5 text-sm text-fg-3">{COVERAGE_NOTE}</p>
      </header>

      {recall.archived && (
        <div className="mt-8">
          <Callout tone="info" title="This notice is archived">
            It was last updated {updated ?? "on an unlisted date"}. Check the official notice for whether it still applies.
          </Callout>
        </div>
      )}

      <div className="mt-10 grid gap-10 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <div className="space-y-8">
          {sections.map(([title, text]) => (
            <section key={title} aria-labelledby={"s-" + title}>
              <h2 id={"s-" + title} className="h3">
                {title}
              </h2>
              <p className={"measure mt-2 whitespace-pre-line leading-relaxed " + (text ? "text-fg-2" : "text-fg-3")}>
                {text ?? "Not listed in this record. See the official notice."}
              </p>
            </section>
          ))}
          {recall.categories.length > 0 && (
            <section aria-labelledby="s-categories">
              <h2 id="s-categories" className="h3">
                Categories
              </h2>
              <ul className="mt-3 flex flex-wrap gap-2">
                {recall.categories.map((category) => (
                  <li key={category} className="inline-flex min-h-8 items-center rounded-[6px] border border-line px-2.5 text-sm text-fg-2">
                    {category}
                  </li>
                ))}
              </ul>
            </section>
          )}
          {recall.product_family && (
            <section aria-labelledby="s-family" className="border-t border-line pt-6">
              <h2 id="s-family" className="text-sm font-medium text-fg">
                Product group <span className="font-normal text-fg-3">(RecallIQ grouping, not from the notice)</span>
              </h2>
              <p className="mt-1 text-sm text-fg-2">{recall.product_family}</p>
              {recall.title_group_size > 1 && (
                <p className="mt-2 text-sm text-fg-3">
                  {recall.title_group_size - 1} other {recall.title_group_size === 2 ? "notice has" : "notices have"} the same title in this sample.
                </p>
              )}
            </section>
          )}
        </div>
        <aside>
          <Suspense fallback={<div className="skeleton h-40" />}>
            <RecallIdentifierPanel recallId={recall.id} />
          </Suspense>
        </aside>
      </div>
    </article>
  );
}

function RecallSkeleton() {
  return (
    <div className="mt-4 space-y-4" aria-busy="true" aria-label="Loading recall notice">
      <div className="skeleton h-4 w-60" />
      <div className="skeleton h-10 w-4/5" />
      <div className="skeleton h-10 w-3/5" />
      <div className="skeleton mt-8 h-11 w-56" />
    </div>
  );
}
