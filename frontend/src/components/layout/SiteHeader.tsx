"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Wordmark } from "../brand/Wordmark";
import { NAV } from "@/lib/nav";
import { Icon } from "../ui/Icon";

/** Reads the URL, so it renders inside <Suspense>; the fallback is the same header without an active link. */
export function SiteHeader() {
  return <HeaderView pathname={usePathname()} />;
}

export function HeaderView({ pathname }: { pathname: string }) {
  const [open, setOpen] = useState(false);
  const [lastPath, setLastPath] = useState(pathname);

  // Close the mobile menu after navigation (adjusting state during render, per React docs).
  if (lastPath !== pathname) {
    setLastPath(pathname);
    setOpen(false);
  }

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const isActive = (href: string) => pathname !== "" && (pathname === href || pathname.startsWith(href + "/"));

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-canvas/80 backdrop-blur-md supports-[backdrop-filter]:bg-canvas/70">
      <div className="mx-auto flex h-16 max-w-[1360px] items-center justify-between gap-6 px-4 sm:px-6 lg:px-10">
        <Link href="/" className="-mx-1 rounded px-1 py-2" aria-label="RecallIQ home">
          <Wordmark />
        </Link>

        <nav aria-label="Main" className="hidden md:block">
          <ul className="flex items-center gap-1">
            {NAV.map((item) => (
              <li key={item.href}>
                <Link
                  href={item.href}
                  aria-current={isActive(item.href) ? "page" : undefined}
                  className={
                    "relative inline-flex h-10 items-center rounded-full px-4 text-[0.9375rem] transition-colors " +
                    (isActive(item.href) ? "text-fg" : "text-fg-2 hover:text-fg")
                  }
                >
                  {item.label}
                  {isActive(item.href) && (
                    <span aria-hidden className="absolute inset-x-4 -bottom-[13px] h-px bg-accent" />
                  )}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        <button
          type="button"
          className="btn btn-ghost -mr-2 px-3 md:hidden"
          aria-expanded={open}
          aria-controls="mobile-nav"
          onClick={() => setOpen((value) => !value)}
        >
          <Icon name={open ? "cross" : "menu"} />
          <span className="sr-only">{open ? "Close menu" : "Open menu"}</span>
        </button>
      </div>

      <nav
        id="mobile-nav"
        aria-label="Main"
        hidden={!open}
        className="border-t border-line bg-canvas px-4 pb-4 md:hidden"
      >
        <ul>
          {NAV.map((item) => (
            <li key={item.href} className="border-b border-line last:border-0">
              <Link
                href={item.href}
                aria-current={isActive(item.href) ? "page" : undefined}
                className={
                  "flex min-h-14 items-center justify-between text-lg font-display " +
                  (isActive(item.href) ? "text-accent" : "text-fg")
                }
              >
                {item.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}
