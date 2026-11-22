// Field-for-field mirror of backend/app/schemas.py. Keep in sync when the API changes.

export type Domain = "food" | "consumer_product";
export type SearchModel = "bm25" | "minilm" | "minilm_finetuned" | "minilm_hard_negative";
export type ScoreType = "bm25" | "cosine_similarity";
export type IdentifierVerdict = "MATCH" | "MISMATCH" | "UNKNOWN";
export type Confidence = "high" | "medium" | "low";

export interface RecallSummary {
  id: number;
  title: string;
  url: string;
  domain: Domain;
  organization: string;
  category: string | null;
  categories: string[];
  recall_class: string | null;
  last_updated: string | null; // ISO date
  archived: boolean;
  product_family: string | null;
}

export interface RecallDetail extends RecallSummary {
  product: string | null;
  issue: string | null;
  what_you_should_do: string | null;
  title_group: string;
  title_group_size: number;
  family_source: string;
  family_confidence: Confidence;
  needs_review: boolean;
  review_reason: string | null;
}

export interface RecallPage {
  items: RecallSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface CategoryCount {
  category: string;
  domain: Domain;
  count: number;
}

export interface CategoryList {
  items: CategoryCount[];
  total: number;
}

export interface SearchRequest {
  query: string;
  model?: SearchModel;
  top_k?: number;
  domain?: Domain | null;
  category?: string | null;
}

export interface SearchHit {
  rank: number;
  recall_id: number;
  title: string;
  category: string | null;
  issue: string | null;
  url: string;
  score: number;
  domain: Domain;
  last_updated: string | null;
  archived: boolean;
}

export interface IdentifierTypeCheck {
  verdict: IdentifierVerdict;
  query_values: string[];
  evidence_values: string[];
  matched: string[];
}

export interface IdentifierCheck {
  recall_id: number;
  verdict: IdentifierVerdict;
  by_type: Record<string, IdentifierTypeCheck>;
}

export interface SearchResponse {
  query: string;
  model: SearchModel;
  score_type: ScoreType;
  score_description: string;
  top_k: number;
  domain: Domain | null;
  category: string | null;
  results: SearchHit[];
  identifiers_in_query: boolean;
  identifier_checks: IdentifierCheck[];
  notice: string;
}

export interface EvidenceFact {
  rank: number;
  recall_id: number;
  title: string;
  url: string;
  product: string | null;
  issue: string | null;
  category: string | null;
  recall_class: string | null;
  last_updated: string | null;
  identifier_verdict: IdentifierVerdict;
}

export interface GeneratedExplanation {
  generator: string;
  text: string;
  cited_recall_ids: number[];
  fallback_reason: string | null;
}

export interface ExplainResponse {
  query: string;
  model: SearchModel;
  evidence_status: "no_results" | "identifier_mismatch" | "evidence_found";
  retrieved_facts: EvidenceFact[];
  skipped_identifier_mismatches: number;
  explanation: GeneratedExplanation;
  sources: string[];
  disclaimer: string;
}

export interface ModelStatus {
  model: SearchModel;
  available: boolean;
  score_type: ScoreType;
  description: string;
  reason: string | null;
}

export interface ModelList {
  models: ModelStatus[];
}
