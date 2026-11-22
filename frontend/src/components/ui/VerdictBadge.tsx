import type { IdentifierVerdict } from "@/lib/api/types";
import { Icon } from "./Icon";

/** Emphasis follows meaning for the user: MATCH means "act", so it is the strongest. */
export const VERDICT = {
  MATCH: {
    label: "Code in notice",
    icon: "alert" as const,
    cls: "bg-alert-tint text-alert border-alert/50",
    sentence: "Your code appears in this notice. Read the notice and follow its instructions.",
  },
  MISMATCH: {
    label: "Code not listed",
    icon: "cross" as const,
    cls: "text-neutral border-line-strong",
    sentence: "Your code isn't listed in this notice. Check the notice for every affected code.",
  },
  UNKNOWN: {
    label: "Not comparable",
    icon: "dash" as const,
    cls: "text-unknown border-line",
    sentence:
      "Codes could not be compared exactly: one side has none of this type, or they are written in a form that can't be compared safely.",
  },
} satisfies Record<IdentifierVerdict, unknown>;

export function VerdictBadge({ verdict }: { verdict: IdentifierVerdict }) {
  const v = VERDICT[verdict];
  return (
    <span
      className={
        "inline-flex items-center gap-1.5 rounded-[6px] border px-2 py-0.5 text-[0.8125rem] font-medium whitespace-nowrap " +
        v.cls
      }
    >
      <Icon name={v.icon} size={14} />
      {v.label}
    </span>
  );
}
