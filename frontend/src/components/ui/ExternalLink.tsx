import type { ReactNode } from "react";
import { Icon } from "./Icon";

/** Opens an official page in a new tab and says so to screen readers. */
export function ExternalLink({
  href,
  children,
  className = "link",
  icon = true,
}: {
  href: string;
  children: ReactNode;
  className?: string;
  icon?: boolean;
}) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className={className}>
      {children}
      {icon && <Icon name="external" size={15} className="ml-1 inline-block -translate-y-px" />}
      <span className="sr-only"> (opens in a new tab)</span>
    </a>
  );
}
