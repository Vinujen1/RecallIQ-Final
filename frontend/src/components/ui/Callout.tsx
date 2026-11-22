import type { ReactNode } from "react";
import { Icon } from "./Icon";

const TONES = {
  info: { box: "border-info/30 bg-info-tint", icon: "text-info", name: "info" as const },
  danger: { box: "border-danger/35 bg-danger-tint", icon: "text-danger", name: "alert" as const },
  quiet: { box: "border-line bg-transparent", icon: "text-fg-3", name: "info" as const },
};

export function Callout({
  tone = "info",
  title,
  children,
  action,
  role,
}: {
  tone?: keyof typeof TONES;
  title?: string;
  children?: ReactNode;
  action?: ReactNode;
  role?: "alert" | "status";
}) {
  const t = TONES[tone];
  return (
    <div role={role} className={"flex gap-3 rounded-[10px] border px-4 py-3 text-sm " + t.box}>
      <Icon name={t.name} size={18} className={"mt-0.5 shrink-0 " + t.icon} />
      <div className="min-w-0 flex-1 text-fg-2">
        {title && <p className="font-medium text-fg">{title}</p>}
        {children && <div className={title ? "mt-0.5" : ""}>{children}</div>}
        {action && <div className="mt-2.5">{action}</div>}
      </div>
    </div>
  );
}
