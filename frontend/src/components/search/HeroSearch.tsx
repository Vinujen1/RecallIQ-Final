"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { EXAMPLE_QUERIES } from "@/data/examples";
import { searchHref } from "@/lib/search-params";
import { Icon } from "../ui/Icon";

/** Landing search: submits to /search, where the real request happens. */
export function HeroSearch() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [error, setError] = useState(false);

  return (
    <div>
      <form
        role="search"
        aria-label="Search recall notices"
        noValidate
        onSubmit={(event) => {
          event.preventDefault();
          const q = query.trim();
          if (!q) {
            setError(true);
            return;
          }
          router.push(searchHref({ q }));
        }}
        className="group relative flex items-center rounded-full border border-line-strong bg-surface p-1.5 shadow-[var(--shadow-raised)] transition-[border-color,box-shadow] duration-200 focus-within:border-accent focus-within:shadow-[0_0_0_4px_color-mix(in_srgb,var(--accent)_18%,transparent),var(--shadow-raised)]"
      >
        <label htmlFor="hero-q" className="sr-only">
          Describe the product
        </label>
        <Icon name="search" size={20} className="ml-3.5 shrink-0 text-fg-3" />
        <input
          id="hero-q"
          name="q"
          type="search"
          value={query}
          maxLength={500}
          autoComplete="off"
          placeholder="Brand, product, or a code from the package"
          aria-invalid={error || undefined}
          aria-describedby={error ? "hero-q-error" : undefined}
          onChange={(event) => {
            setQuery(event.target.value);
            if (error) setError(false);
          }}
          className="min-h-12 min-w-0 flex-1 bg-transparent px-3 text-base text-fg placeholder:text-fg-3 focus:outline-none [&::-webkit-search-cancel-button]:hidden"
        />
        <button type="submit" className="btn btn-primary rounded-full px-5 sm:px-6">
          Search
        </button>
      </form>
      <p id="hero-q-error" role="alert" className="mt-2 min-h-5 pl-5 text-sm text-danger">
        {error ? "Describe a product first, such as its brand or what it is." : ""}
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-sm">
        <span className="text-fg-3">Try</span>
        {EXAMPLE_QUERIES.map((example) => (
          <Link
            key={example}
            href={searchHref({ q: example })}
            className="inline-flex min-h-9 items-center rounded-full border border-line px-3.5 text-fg-2 transition-colors hover:border-accent hover:text-fg"
          >
            {example}
          </Link>
        ))}
      </div>
    </div>
  );
}
