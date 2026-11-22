// Generates src/data/model-lab.json from the committed experiment artifacts.
// Every number shown in /model-lab comes from here; nothing is typed by hand.
// Usage (from frontend/): node scripts/export-model-lab.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const read = (p) => JSON.parse(readFileSync(join(root, p), "utf-8"));
const need = (obj, key, where) => {
  if (obj == null || !(key in obj)) throw new Error(`Missing "${key}" in ${where}`);
  return obj[key];
};

const proxy = read("experiments/runs/proxy_validation.json");
const results = need(proxy, "results", "proxy_validation.json");
const SYSTEMS = [
  ["bm25", "bm25"],
  ["minilm", "minilm"],
  ["model_b", "minilm_finetuned"],
  ["model_c", "minilm_hard_negative"],
];

const systems = SYSTEMS.map(([key, apiModel]) => {
  const r = need(results, key, "results");
  const mean = need(r, "mean", key);
  const fmr = need(r, "false_match_rate@1", key);
  const byView = Object.fromEntries(
    Object.entries(need(r, "by_view", key)).map(([view, v]) => [
      view.replace(/^pseudo:/, ""),
      { queries: v.num_queries, mrr10: v["mrr@10"], recall5: v["recall@5"] },
    ]),
  );
  return {
    key,
    apiModel,
    recall5: need(mean, "recall@5", key),
    mrr10: need(mean, "mrr@10", key),
    ndcg10: need(mean, "ndcg@10", key),
    fmr1: { value: fmr.value, failures: fmr.numerator, queries: fmr.denominator },
    byView,
  };
});

const comparisons = Object.entries(need(proxy, "comparisons", "proxy_validation.json")).map(
  ([name, c]) => {
    const b = need(c, "bootstrap_mrr@10", name);
    const [treatment, control] = name.split("_vs_");
    if (!treatment || !control) throw new Error(`Unexpected comparison name ${name}`);
    return {
      name,
      treatment,
      control,
      meanDifference: b.mean_difference,
      ci95: b.ci95,
      excludesZero: b.ci_excludes_zero,
      better: b.better,
      worse: b.worse,
      ties: b.ties,
      resamples: b.resamples,
      seed: b.seed,
    };
  },
);

const training = Object.fromEntries(
  ["model-b-v1", "model-c-v1"].map((run) => {
    const s = read(`experiments/runs/${run}/summary.json`);
    const h = read(`experiments/runs/${run}/history.json`);
    return [
      run,
      {
        label: need(s, "experiment_label", run),
        bestEpoch: need(s, "best_epoch", run),
        epochsRun: need(s, "epochs_run", run),
        stoppedEarly: need(s, "stopped_early", run),
        wallClockSeconds: need(s, "wall_clock_seconds", run),
        epochs: need(h, "epochs", run).map((e) => ({
          epoch: e.epoch,
          trainLoss: e.train_loss,
          validationLoss: e.validation_loss,
          proxyMrr10: e.proxy?.["mrr@10"] ?? null,
        })),
      },
    ];
  }),
);

// Multi-seed v2 runs (training artifacts are git-ignored; seeds is null without them).
// Same values as experiments/reports/seeds.md: the selected (best-epoch) checkpoint's proxy.
const SEEDS = [4110, 4111, 4112];
let seeds = null;
try {
  const runs = [];
  for (const model of ["b", "c"]) {
    for (const seed of SEEDS) {
      const dir = `ml/artifacts/models/model-${model}-v2-s${seed}`;
      const s = read(`${dir}/summary.json`);
      const h = read(`${dir}/history.json`);
      const best = need(h, "epochs", dir).find((e) => e.epoch === need(s, "best_epoch", dir));
      if (!best?.proxy) throw new Error(`No proxy metrics for best epoch in ${dir}`);
      runs.push({
        model: model.toUpperCase(),
        seed,
        bestEpoch: s.best_epoch,
        epochsRun: s.epochs_run,
        recall5: best.proxy["recall@5"],
        mrr10: best.proxy["mrr@10"],
        ndcg10: best.proxy["ndcg@10"],
      });
    }
  }
  seeds = { seeds: SEEDS, runs };
} catch (error) {
  console.warn(`Multi-seed runs not exported: ${error.message}`);
}

const prov = need(proxy, "provenance", "proxy_validation.json");
const out = {
  generatedFrom: "experiments/runs (proxy_validation.json, model-*-v1)",
  label: need(proxy, "label", "proxy_validation.json"),
  queries: need(proxy, "num_queries", "proxy_validation.json"),
  corpusSize: need(proxy, "corpus_size", "proxy_validation.json"),
  evaluatedAt: prov.generated_at,
  splitFrozen: prov.split_frozen,
  splitFingerprint: String(prov.split_assignment_fingerprint).slice(0, 12),
  corpusFingerprint: String(prov.corpus_fingerprint).slice(0, 12),
  systems,
  comparisons,
  training,
  seeds,
};
writeFileSync(join(root, "frontend/src/data/model-lab.json"), JSON.stringify(out, null, 2) + "\n");
console.log(`model-lab.json: ${systems.length} systems, ${comparisons.length} comparisons`);
