import type { IdentifierCheck } from "@/lib/api/types";
import { VERDICT, VerdictBadge } from "../ui/VerdictBadge";

const TYPE_LABEL: Record<string, string> = {
  upc: "UPC",
  model: "Model",
  lot: "Lot",
  best_before: "Best before",
};

/** Each code on one unbroken line; the list wraps between codes, never inside one. */
function Values({ values }: { values: string[] }) {
  if (!values.length) return <span className="text-fg-3">none</span>;
  return (
    <span className="inline-flex flex-wrap gap-x-2 gap-y-0.5 align-top">
      {values.map((value) => (
        <span key={value} className="num whitespace-nowrap text-fg [overflow-wrap:anywhere]">
          {value}
        </span>
      ))}
    </span>
  );
}

/** Both sides of the deterministic identifier comparison, one row per identifier type. */
export function IdentifierChecks({ check }: { check: IdentifierCheck }) {
  const rows = Object.entries(check.by_type).filter(
    ([, t]) => t.query_values.length > 0 || t.evidence_values.length > 0,
  );
  const bothSides = rows.some(([, t]) => t.query_values.length > 0 && t.evidence_values.length > 0);
  // The sentence follows the rows shown, so it never contradicts them.
  const sentence =
    check.verdict === "UNKNOWN" && bothSides
      ? "Your code and the notice's codes are written differently, so they could not be compared exactly. Check the official notice."
      : VERDICT[check.verdict].sentence;

  return (
    <div className="space-y-3">
      <VerdictBadge verdict={check.verdict} />
      <p className="text-sm text-fg-2">{sentence}</p>
      {rows.length > 0 && (
        <dl className="divide-y divide-line rounded-[10px] border border-line text-sm">
          {rows.map(([type, t]) => (
            <div key={type} className="grid gap-x-4 gap-y-1 px-3.5 py-3 sm:grid-cols-[6.5rem_1fr]">
              <dt className="font-medium text-fg">{TYPE_LABEL[type] ?? type}</dt>
              <dd className="min-w-0 space-y-1">
                <p>
                  <span className="text-fg-3">You entered: </span>
                  <Values values={t.query_values} />
                </p>
                <p>
                  <span className="text-fg-3">Notice lists: </span>
                  <Values values={t.evidence_values} />
                </p>
                {t.matched.length > 0 && (
                  <p>
                    <span className="text-fg-3">Same in both: </span>
                    <span className="num font-medium text-alert">{t.matched.join(", ")}</span>
                  </p>
                )}
              </dd>
            </div>
          ))}
        </dl>
      )}
      <p className="text-[0.8125rem] text-fg-3">
        Codes are shown normalised (spaces and punctuation removed) as stored in RecallIQ&apos;s copy of the
        notice. The official notice shows them as written and may list more.
      </p>
    </div>
  );
}
