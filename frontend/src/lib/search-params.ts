import type { Domain, SearchModel } from "./api/types";
import { DEFAULT_MODEL, isSearchModel } from "./models";

export interface SearchState {
  q: string;
  model: SearchModel;
  domain: Domain | null;
  cat: string | null;
  id: number | null;
}

type Readable = { get(name: string): string | null };

export function readSearchState(params: Readable): SearchState {
  const model = params.get("model");
  const domain = params.get("domain");
  const id = Number(params.get("id"));
  return {
    q: (params.get("q") ?? "").trim().slice(0, 500),
    model: isSearchModel(model) ? model : DEFAULT_MODEL,
    domain: domain === "food" || domain === "consumer_product" ? domain : null,
    cat: params.get("cat")?.trim().slice(0, 200) || null,
    id: Number.isSafeInteger(id) && id > 0 ? id : null,
  };
}

/** Query string for a search state; defaults are omitted so shared links stay short. */
export function searchQuery(state: Partial<SearchState>): string {
  const params = new URLSearchParams();
  if (state.q) params.set("q", state.q);
  if (state.model && state.model !== DEFAULT_MODEL) params.set("model", state.model);
  if (state.domain) params.set("domain", state.domain);
  if (state.cat) params.set("cat", state.cat);
  if (state.id) params.set("id", String(state.id));
  return params.toString();
}

export function searchHref(state: Partial<SearchState>): string {
  const query = searchQuery(state);
  return query ? "/search?" + query : "/search";
}

/** Link to a recall's full page that remembers the search it came from. */
export function recallHref(id: number, from?: Partial<SearchState>): string {
  const query = from ? searchQuery({ ...from, id }) : "";
  return "/recalls/" + id + (query ? "?" + query : "");
}
