import { NextResponse, type NextRequest } from "next/server";
import { API_BASE_URL } from "@/lib/api/client";

/**
 * Real 404 status for missing recall pages.
 *
 * With Cache Components the recall page streams its prerendered shell first, so a
 * notFound() inside the page can only render the not-found UI with a 200 status. The
 * Next.js docs recommend checking the slug in Proxy, before streaming starts. This check
 * is a single small lookup (GET /recalls/{id}, one SQLite row) with a short timeout.
 *
 * Missing or malformed ids are rewritten to an unmatched internal path, so Next serves the
 * app's own not-found.tsx with status 404. If the API is unreachable, slow or errors, the
 * request passes through and the page shows its normal error state; Proxy never invents a 404.
 */
const MISSING = "/__recall-not-found";
const CHECK_TIMEOUT_MS = 1500;

export async function proxy(request: NextRequest) {
  const raw = request.nextUrl.pathname.split("/")[2] ?? "";
  const id = /^[1-9]\d{0,17}$/.test(raw) ? Number(raw) : NaN;
  if (!Number.isSafeInteger(id)) {
    return NextResponse.rewrite(new URL(MISSING, request.url));
  }
  try {
    const response = await fetch(API_BASE_URL + "/recalls/" + id, {
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(CHECK_TIMEOUT_MS),
      cache: "no-store",
    });
    if (response.status === 404) {
      return NextResponse.rewrite(new URL(MISSING, request.url));
    }
  } catch {
    // API down or slow: let the page render and report it.
  }
  return NextResponse.next();
}

export const config = {
  matcher: "/recalls/:id",
};
