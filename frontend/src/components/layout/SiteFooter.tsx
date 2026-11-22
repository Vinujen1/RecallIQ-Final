import Link from "next/link";
import { OFFICIAL_SITE } from "@/lib/format";
import { Wordmark } from "../brand/Wordmark";
import { ExternalLink } from "../ui/ExternalLink";
import { NAV } from "@/lib/nav";
import { ThemeToggle } from "./ThemeToggle";

export function SiteFooter() {
  return (
    <footer className="mt-auto border-t border-line">
      <div className="mx-auto grid max-w-[1360px] gap-10 px-4 py-12 sm:px-6 md:grid-cols-[1.4fr_1fr_1fr] lg:px-10">
        <div className="max-w-sm space-y-4">
          <Wordmark />
          <p className="text-sm leading-relaxed text-fg-2">
            A course project that finds the official Government of Canada recall notices closest to an
            everyday product description. It is not a government service and does not decide whether your product is
            recalled.
          </p>
        </div>

        <nav aria-label="Footer">
          <h2 className="mb-3 text-sm font-medium text-fg">Explore</h2>
          <ul className="space-y-1 text-sm">
            {[{ href: "/", label: "Home" }, ...NAV].map((item) => (
              <li key={item.href}>
                <Link href={item.href} className="inline-flex min-h-11 items-center text-fg-2 hover:text-fg md:min-h-8">
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        <div className="space-y-3 text-sm">
          <h2 className="font-medium text-fg">Data source</h2>
          <p className="text-fg-2">
            Recall notices from{" "}
            <ExternalLink href={OFFICIAL_SITE}>Recalls and safety alerts, Government of Canada</ExternalLink>.
            English only; official notices are also available in French.
          </p>
        </div>
      </div>
      <div className="border-t border-line">
        <div className="mx-auto flex max-w-[1360px] flex-col gap-4 px-4 py-5 text-[0.8125rem] text-fg-3 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-10">
          <p>RecallIQ, ARTE 4110U Group 5. An empty result does not mean a product is safe.</p>
          <ThemeToggle />
        </div>
      </div>
    </footer>
  );
}
