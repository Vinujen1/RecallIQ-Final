"use client";

import type { MouseEvent } from "react";
import type { IdentifierCheck, SearchHit } from "@/lib/api/types";
import { DOMAIN_LABEL, formatDate } from "@/lib/format";
import { VerdictBadge } from "../ui/VerdictBadge";
import { IdentifierChecks } from "./IdentifierChecks";

/**
 * A ranked notice. It is a real link to the full recall page (so open-in-new-tab and
 * small screens work); on wide screens `onSelect` intercepts it to fill the side panel.
 */
export function ResultRow({
  hit,
  href,
  check,
  showCheck,
  selected,
  onSelect,
}: {
  hit: SearchHit;
  href: string;
  check?: IdentifierCheck;
  showCheck: boolean;
  selected: boolean;
  onSelect: (event: MouseEvent<HTMLAnchorElement>, id: number) => void;
}) {
  const updated = formatDate(hit.last_updated);
  return (
    <li
      className={
        "relative border-b border-line transition-colors duration-200 last:border-0 " +
        (selected ? "bg-raised lg:before:absolute lg:before:inset-y-3 lg:before:left-0 lg:before:w-0.5 lg:before:rounded-full lg:before:bg-accent" : "hover:bg-surface")
      }
    >
      <a
        href={href}
        aria-current={selected ? "true" : undefined}
        onClick={(event) => onSelect(event, hit.recall_id)}
        className="grid grid-cols-[2rem_1fr] gap-x-3 px-4 py-4 sm:px-5 focus-visible:-outline-offset-2"
      >
        <span className="num pt-0.5 text-sm text-fg-3" aria-label={"Rank " + hit.rank}>
          {hit.rank}
        </span>
        <span className="min-w-0">
          <span className="line-clamp-2 font-medium leading-snug text-fg">{hit.title}</span>
          {hit.issue && <span className="mt-1 block line-clamp-1 text-sm text-fg-2">{hit.issue}</span>}
          <span className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[0.8125rem] text-fg-3">
            <span>
              Updated <span className="tnum text-fg-2">{updated ?? "date not listed"}</span>
            </span>
            {hit.archived && (
              <span className="rounded-[6px] border border-line-strong px-1.5 py-px text-fg-2">Archived</span>
            )}
            <span>{hit.category ?? DOMAIN_LABEL[hit.domain]}</span>
          </span>
          {showCheck && check && (
            <span className="mt-2.5 block">
              <VerdictBadge verdict={check.verdict} />
            </span>
          )}
        </span>
      </a>
      {showCheck && check && check.verdict !== "UNKNOWN" && (
        <details className="-mt-3 px-4 pb-3 pl-[3.75rem] sm:pl-[4rem] lg:hidden">
          <summary className="inline-flex min-h-11 cursor-pointer items-center text-sm text-accent">
            Compare codes
          </summary>
          <div className="mt-1 pb-1">
            <IdentifierChecks check={check} />
          </div>
        </details>
      )}
    </li>
  );
}
