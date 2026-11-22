"use client";

import Link from "next/link";
import { errorMessage } from "@/lib/api/client";

export default function Error({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  // Server errors arrive without their original class; show the API wording when it survives.
  const message = error.name === "ApiError" ? errorMessage(error) : "Something went wrong while loading this page.";
  return (
    <div className="mx-auto w-full max-w-[1100px] px-4 py-20 sm:px-6 lg:px-10" role="alert">
      <h1 className="h1 max-w-2xl">This page didn&apos;t load</h1>
      <p className="mt-4 max-w-xl text-fg-2">{message} If the RecallIQ API is not running, start it and try again.</p>
      <div className="mt-8 flex flex-wrap gap-3">
        <button type="button" onClick={reset} className="btn btn-primary">
          Try again
        </button>
        <Link href="/" className="btn btn-secondary">
          Go to the home page
        </Link>
      </div>
    </div>
  );
}
