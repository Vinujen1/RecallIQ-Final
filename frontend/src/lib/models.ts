import type { SearchModel } from "./api/types";

export interface ModelInfo {
  key: SearchModel;
  name: string;
  short: string;
  method: string;
  /** Completes "ranked by ..." */
  ranking: string;
}

/** Fixed mapping used across the site: A = pretrained, B = fine-tuned, C = hard negatives. */
export const MODELS: ModelInfo[] = [
  {
    key: "bm25",
    name: "BM25",
    short: "Keyword baseline",
    method: "Okapi BM25 over each notice's title, product, issue and category. No training.",
    ranking: "shared keywords",
  },
  {
    key: "minilm",
    name: "Model A",
    short: "Pretrained MiniLM",
    method: "all-MiniLM-L6-v2 sentence embeddings as published, compared by cosine similarity.",
    ranking: "meaning, using Model A",
  },
  {
    key: "minilm_finetuned",
    name: "Model B",
    short: "Fine-tuned MiniLM",
    method: "MiniLM fine-tuned on recall notices with a contrastive loss and in-batch negatives.",
    ranking: "meaning, using Model B",
  },
  {
    key: "minilm_hard_negative",
    name: "Model C",
    short: "Hard-negative MiniLM",
    method: "Model B's recipe plus mined and human-verified look-alike notices as hard negatives.",
    ranking: "meaning, using Model C",
  },
];

/** BM25 is the public default while independent human-reviewed evaluation is pending. */
export const DEFAULT_MODEL: SearchModel = "bm25";

export const modelInfo = (key: SearchModel): ModelInfo => MODELS.find((m) => m.key === key) ?? MODELS[0];

export const isSearchModel = (value: string | null): value is SearchModel => MODELS.some((m) => m.key === value);
