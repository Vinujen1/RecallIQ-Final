import Link from "next/link";
import { ExternalLink } from "@/components/ui/ExternalLink";
import { OFFICIAL_SEARCH_URL } from "@/lib/format";

export default function NotFound() {
  return (
    <div className="mx-auto w-full max-w-[1100px] px-4 py-20 sm:px-6 lg:px-10">
      <p className="num text-sm text-accent">404</p>
      <h1 className="h1 mt-3 max-w-2xl">This page or recall notice isn&apos;t in RecallIQ</h1>
      <p className="mt-4 max-w-xl text-fg-2">
        RecallIQ holds a 1,000-notice sample, so a notice can exist on the official site without being here.
      </p>
      <div className="mt-8 flex flex-wrap gap-3">
        <Link href="/search" className="btn btn-primary">
          Search recall notices
        </Link>
        <ExternalLink href={OFFICIAL_SEARCH_URL} className="btn btn-secondary">
          Search all official recalls
        </ExternalLink>
      </div>
    </div>
  );
}
