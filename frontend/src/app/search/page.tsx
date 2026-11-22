import type { Metadata } from "next";
import { Suspense } from "react";
import { SearchApp } from "@/components/search/SearchApp";

export const metadata: Metadata = {
  title: "Search",
  description: "Find official Government of Canada recall notices that match a product description or code.",
};

export default function SearchPage() {
  return (
    // useSearchParams needs a Suspense boundary under cacheComponents; the shell prerenders.
    <Suspense fallback={<div className="mx-auto w-full max-w-[1360px] px-4 pt-10 sm:px-6 lg:px-10" aria-busy="true"><div className="skeleton h-12" /></div>}>
      <SearchApp />
    </Suspense>
  );
}
