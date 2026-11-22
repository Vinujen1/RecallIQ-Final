import type {
  CategoryList,
  ExplainResponse,
  ModelList,
  RecallDetail,
  RecallPage,
  SearchRequest,
  SearchResponse,
} from "./types";

/** Public API base URL (browser and server). A URL only: never put secrets in NEXT_PUBLIC_*. */
export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type ApiErrorKind = "network" | "timeout" | "not_found" | "invalid" | "unavailable" | "server";

export class ApiError extends Error {
  constructor(
    public readonly kind: ApiErrorKind,
    public readonly status: number | null,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const MESSAGES: Record<ApiErrorKind, string> = {
  network: "Can't reach the RecallIQ API. Check that the backend is running, then try again.",
  timeout: "The RecallIQ API took too long to respond. Try again.",
  not_found: "That recall notice isn't in the RecallIQ index.",
  invalid: "The request wasn't accepted. Check the search text and try again.",
  unavailable: "This retrieval model is unavailable on this server. Choose another model.",
  server: "The RecallIQ API returned an error. Try again in a moment.",
};

export function errorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : MESSAGES.server;
}

export const isAbort = (error: unknown) => error instanceof DOMException && error.name === "AbortError";

function kindFor(status: number): ApiErrorKind {
  if (status === 404) return "not_found";
  if (status === 422 || status === 400) return "invalid";
  if (status === 503) return "unavailable";
  return "server";
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 20000): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const outer = init.signal;
  const forward = () => controller.abort();
  if (outer?.aborted) controller.abort();
  outer?.addEventListener("abort", forward);
  let response: Response;
  try {
    response = await fetch(API_BASE_URL + path, {
      ...init,
      signal: controller.signal,
      headers: { Accept: "application/json", ...(init.body ? { "Content-Type": "application/json" } : {}) },
    });
  } catch {
    if (outer?.aborted) throw new DOMException("Request cancelled", "AbortError");
    const kind: ApiErrorKind = timedOut ? "timeout" : "network";
    throw new ApiError(kind, null, MESSAGES[kind]);
  } finally {
    clearTimeout(timer);
    outer?.removeEventListener("abort", forward);
  }
  if (!response.ok) {
    // The server's `detail` text is deliberately not shown: it can contain server paths.
    const kind = kindFor(response.status);
    throw new ApiError(kind, response.status, MESSAGES[kind]);
  }
  return (await response.json()) as T;
}

export const EXPLAIN_TIMEOUT_MS = 45000;

const post = (body: SearchRequest, signal?: AbortSignal): RequestInit => ({
  method: "POST",
  body: JSON.stringify(body),
  signal,
});

export const api = {
  search: (body: SearchRequest, signal?: AbortSignal) => request<SearchResponse>("/search", post(body, signal)),
  // Longer timeout: the server may call an LLM (bounded by RECALLIQ_LLM_TIMEOUT_SECONDS + retries).
  explain: (body: SearchRequest, signal?: AbortSignal) =>
    request<ExplainResponse>("/explain", post(body, signal), EXPLAIN_TIMEOUT_MS),
  models: (signal?: AbortSignal) => request<ModelList>("/search/models", { signal }),
  recall: (id: number, signal?: AbortSignal) => request<RecallDetail>("/recalls/" + id, { signal }),
  recent: (limit = 5, signal?: AbortSignal) => request<RecallPage>("/recalls?limit=" + limit, { signal }),
  categories: (signal?: AbortSignal) => request<CategoryList>("/categories", { signal }),
};
