import type { Metadata } from "next";
import { CompareModels } from "@/components/lab/CompareModels";
import { DifferencePlot, type Difference } from "@/components/lab/DifferencePlot";
import { TrainingCurve } from "@/components/lab/TrainingCurve";
import lab from "@/data/model-lab.json";
import { fixed, signed } from "@/lib/format";
import { ModelAvailability } from "@/components/lab/ModelAvailability";
import { MODELS } from "@/lib/models";

export const metadata: Metadata = {
  title: "Model Lab",
  description: "How RecallIQ's four retrieval approaches were built and how they compare on proxy evaluation.",
};

// Every number on this page comes from src/data/model-lab.json, which
// scripts/export-model-lab.mjs generates from the committed experiment files. Nothing here
// is typed by hand, so the page cannot drift from experiments/reports.
const NAME: Record<string, string> = { bm25: "BM25", minilm: "Model A", model_b: "Model B", model_c: "Model C" };

// Used to highlight the best value per column (lower is better for the false-match rate).
const best = (key: "recall5" | "mrr10" | "ndcg10") => Math.max(...lab.systems.map((s) => s[key]));
const bestFmr = Math.min(...lab.systems.map((s) => s.fmr1.value));

const differences: Difference[] = lab.comparisons.map((c) => ({
  label: `${NAME[c.treatment]} vs ${NAME[c.control]}`,
  mean: c.meanDifference,
  low: c.ci95[0],
  high: c.ci95[1],
  better: c.better,
  worse: c.worse,
}));

const curves = [
  { run: "model-b-v1", name: "Model B", color: "var(--series-b)", dashed: true, marker: "square" as const },
  { run: "model-c-v1", name: "Model C", color: "var(--series-c)", dashed: false, marker: "circle" as const },
].map(({ run, ...rest }) => {
  const t = lab.training[run as keyof typeof lab.training];
  return {
    ...rest,
    bestEpoch: t.bestEpoch,
    seconds: t.wallClockSeconds,
    points: t.epochs.filter((e) => e.proxyMrr10 !== null).map((e) => ({ epoch: e.epoch, value: e.proxyMrr10 as number })),
  };
});

function Mark({ on }: { on: boolean }) {
  return on ? <span className="ml-1.5 text-accent" title="Best in column">(best)</span> : null;
}

export default function ModelLabPage() {
  const seedRuns = lab.seeds?.runs ?? [];
  const seedPairs = (lab.seeds?.seeds ?? []).map((seed) => {
    const b = seedRuns.find((r) => r.model === "B" && r.seed === seed)!;
    const c = seedRuns.find((r) => r.model === "C" && r.seed === seed)!;
    return { seed, b: b.mrr10, c: c.mrr10, diff: c.mrr10 - b.mrr10 };
  });
  const meanDiff = seedPairs.reduce((sum, p) => sum + p.diff, 0) / (seedPairs.length || 1);

  return (
    <div className="mx-auto w-full max-w-[1200px] px-4 pb-24 pt-10 sm:px-6 lg:px-10 lg:pt-14">
      <header className="max-w-3xl">
        <p className="kicker">Model Lab</p>
        <h1 className="h1 mt-3">Four ways to rank a recall notice, and what we can honestly say about them</h1>
        <p className="mt-5 text-lg leading-relaxed text-fg-2">
          RecallIQ can rank notices with a keyword baseline or with one of three MiniLM language models. All four
          search the same {lab.corpusSize.toLocaleString("en-CA")} notices. Public search uses BM25 by default while
          independent, human-reviewed evaluation is still pending.
        </p>
      </header>

      <div className="mt-8 rounded-[16px] border border-info/35 bg-info-tint p-5 sm:p-6">
        <p className="font-medium text-fg">
          <span className="mr-2 inline-flex rounded-[6px] border border-info/60 px-1.5 py-px text-[0.8125rem] font-semibold text-info">Proxy</span>
          These are proxy results, not human judgments
        </p>
        <p className="mt-2 max-w-3xl text-fg-2">
          Every number below comes from {lab.queries} validation queries generated automatically from notice text, each
          scored against the notice it was generated from. They are useful for comparing training runs, but they are
          not independent human evaluation, and the held-out test split has not been evaluated.
        </p>
      </div>

      <section aria-labelledby="approaches" className="mt-16">
        <h2 id="approaches" className="h2">The four approaches</h2>
        <ul className="mt-6 divide-y divide-line border-y border-line">
          {MODELS.map((m) => (
            <li key={m.key} className="grid gap-2 py-5 sm:grid-cols-[12rem_1fr] sm:gap-8">
              <div>
                <p className="font-medium">{m.name}</p>
                <p className="text-sm text-fg-3">{m.short}</p>
                <ModelAvailability model={m.key} />
              </div>
              <p className="text-fg-2">{m.method}</p>
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="results" className="mt-16">
        <h2 id="results" className="h2">Proxy results on {lab.queries} generated queries</h2>
        <p className="mt-3 max-w-3xl text-fg-2">
          Higher is better for Recall@5, MRR@10 and nDCG@10. Lower is better for the top-1 miss rate: the share of queries
          whose first-ranked notice is not the one the query was generated from.
        </p>
        <div className="mt-6 overflow-x-auto rounded-[16px] border border-line">
          <table className="w-full min-w-[640px] text-left text-sm">
            <caption className="sr-only">Proxy retrieval metrics for the four approaches</caption>
            <thead className="border-b border-line text-fg-3">
              <tr>
                <th scope="col" className="px-5 py-3 font-medium">Approach</th>
                <th scope="col" className="px-5 py-3 text-right font-medium">Recall@5</th>
                <th scope="col" className="px-5 py-3 text-right font-medium">MRR@10</th>
                <th scope="col" className="px-5 py-3 text-right font-medium">nDCG@10</th>
                <th scope="col" className="px-5 py-3 text-right font-medium">Top-1 miss rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {lab.systems.map((s) => (
                <tr key={s.key}>
                  <th scope="row" className="px-5 py-3.5 font-medium">{NAME[s.key]}</th>
                  <td className="num px-5 py-3.5 text-right">{fixed(s.recall5)}<Mark on={s.recall5 === best("recall5")} /></td>
                  <td className="num px-5 py-3.5 text-right">{fixed(s.mrr10)}<Mark on={s.mrr10 === best("mrr10")} /></td>
                  <td className="num px-5 py-3.5 text-right">{fixed(s.ndcg10)}<Mark on={s.ndcg10 === best("ndcg10")} /></td>
                  <td className="num px-5 py-3.5 text-right">
                    {fixed(s.fmr1.value)} <span className="text-fg-3">({s.fmr1.failures}/{s.fmr1.queries})</span>
                    <Mark on={s.fmr1.value === bestFmr} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-sm text-fg-3">
          Evaluated {lab.evaluatedAt.slice(0, 10)} on the frozen validation split (split {lab.splitFingerprint}, corpus{" "}
          {lab.corpusFingerprint}). BM25 already scores above 0.95 here, so the proxy is close to its ceiling.
        </p>
      </section>

      <section aria-labelledby="differences" className="mt-16 grid gap-10 lg:grid-cols-[1fr_1.35fr]">
        <div>
          <h2 id="differences" className="h2">Are the differences real?</h2>
          <p className="mt-3 text-fg-2">
            Each line is a paired bootstrap ({lab.comparisons[0].resamples.toLocaleString("en-CA")} resamples) of the difference in
            MRR@10 over the same queries. Only when a whole line sits on one side of zero is the difference distinguishable from
            noise on this proxy.
          </p>
          <p className="mt-3 text-fg-2">
            Fine-tuning clearly improved on the pretrained model (Model B vs Model A). Model C vs Model B, and either fine-tuned
            model vs BM25, overlap zero: on this evidence they are not distinguishable.
          </p>
        </div>
        <div className="rounded-[16px] border border-line bg-surface p-5">
          <div className="hidden sm:block">
            <DifferencePlot rows={differences} />
          </div>
          <div className="text-sm sm:hidden">
            <DifferenceTable />
          </div>
          <details className="mt-3 hidden text-sm sm:block">
            <summary className="inline-flex min-h-11 cursor-pointer items-center text-fg-2 hover:text-fg">Show as a table</summary>
            <DifferenceTable />
          </details>
        </div>
      </section>

      <section aria-labelledby="training" className="mt-16 grid gap-10 lg:grid-cols-[1fr_1.35fr]">
        <div>
          <h2 id="training" className="h2">Training runs</h2>
          <p className="mt-3 text-fg-2">
            Models B and C share every setting except hard negatives: same data, seed, learning rate and validation queries.
            The checkpoint served is the epoch with the best proxy MRR@10. Both ran on a CPU.
          </p>
          <dl className="mt-5 divide-y divide-line border-y border-line text-sm">
            {curves.map((c) => (
              <div key={c.name} className="flex justify-between gap-4 py-3">
                <dt className="text-fg-2">{c.name}</dt>
                <dd className="num text-fg">epoch {c.bestEpoch}, {(c.seconds / 60).toFixed(1)} min</dd>
              </div>
            ))}
          </dl>
        </div>
        <div className="rounded-[16px] border border-line bg-surface p-5">
          <div className="hidden sm:block">
            <TrainingCurve series={curves} best={Object.fromEntries(curves.map((c) => [c.name, c.bestEpoch]))} />
          </div>
          <div className="text-sm sm:hidden">
            <p className="mb-2 text-fg-3">Proxy MRR@10 by epoch (0 = before fine-tuning)</p>
            <CurveTable />
          </div>
          <details className="mt-3 hidden text-sm sm:block">
            <summary className="inline-flex min-h-11 cursor-pointer items-center text-fg-2 hover:text-fg">Show as a table</summary>
            <CurveTable />
          </details>
        </div>
      </section>

      {seedPairs.length > 0 && (
        <section aria-labelledby="seeds" className="mt-16">
          <h2 id="seeds" className="h2">Repeated with three seeds</h2>
          <p className="mt-3 max-w-3xl text-fg-2">
            B and C were retrained with seeds {lab.seeds!.seeds.join(", ")} (with a duplicate-query masking fix). Model C
            scored higher on proxy MRR@10 for every seed, but by a mean of only {signed(meanDiff, 4)}, on a proxy near its
            ceiling. That is a consistent small direction, not proof that C is better for real users.
          </p>
          <div className="mt-6 overflow-x-auto rounded-[16px] border border-line">
            <table className="w-full min-w-[480px] text-left text-sm">
              <caption className="sr-only">Proxy MRR@10 per seed for Models B and C</caption>
              <thead className="border-b border-line text-fg-3">
                <tr>
                  <th scope="col" className="px-5 py-3 font-medium">Seed</th>
                  <th scope="col" className="px-5 py-3 text-right font-medium">Model B MRR@10</th>
                  <th scope="col" className="px-5 py-3 text-right font-medium">Model C MRR@10</th>
                  <th scope="col" className="px-5 py-3 text-right font-medium">C minus B</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {seedPairs.map((p) => (
                  <tr key={p.seed}>
                    <th scope="row" className="num px-5 py-3 font-normal">{p.seed}</th>
                    <td className="num px-5 py-3 text-right">{p.b.toFixed(4)}</td>
                    <td className="num px-5 py-3 text-right">{p.c.toFixed(4)}</td>
                    <td className="num px-5 py-3 text-right">{signed(p.diff, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section aria-labelledby="compare" className="mt-16">
        <h2 id="compare" className="h2">Try a query on all four</h2>
        <p className="mt-3 max-w-3xl text-fg-2">
          Runs the same search with each approach and lists the top five. Only ranks are shown: a BM25 score and a cosine
          similarity measure different things and cannot be compared.
        </p>
        <div className="mt-6">
          <CompareModels />
        </div>
      </section>

      <section aria-labelledby="limits" className="mt-16 border-t border-line pt-10">
        <h2 id="limits" className="h2">What this evaluation cannot show</h2>
        <ul className="mt-5 grid gap-x-10 gap-y-4 text-fg-2 md:grid-cols-2">
          <li>Generated queries are easier and noisier than real ones. Some describe several notices at once, which counts as a miss.</li>
          <li>The proxy is near its ceiling, so small differences between strong systems are hard to see.</li>
          <li>Independent, human-written validation and test queries are still being written and cross-reviewed.</li>
          <li>Identifier precision can&apos;t be measured yet: only 6 of the 1,000 notices list a code in their stored text.</li>
        </ul>
      </section>
    </div>
  );
}

function DifferenceTable() {
  return (
    <table className="mt-2 w-full text-left text-sm">
      <thead className="text-fg-3">
        <tr>
          <th scope="col" className="py-2 font-medium">Comparison</th>
          <th scope="col" className="py-2 text-right font-medium">Mean</th>
          <th scope="col" className="py-2 text-right font-medium">95% interval</th>
          <th scope="col" className="py-2 text-right font-medium">Better / worse</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-line">
        {differences.map((d) => (
          <tr key={d.label}>
            <th scope="row" className="py-2 font-normal">{d.label}</th>
            <td className="num py-2 text-right">{signed(d.mean, 4)}</td>
            <td className="num py-2 text-right">{signed(d.low, 4)} to {signed(d.high, 4)}</td>
            <td className="num py-2 text-right">{d.better} / {d.worse}</td>
          </tr>
        ))}
      </tbody>
    </table>  );
}

function CurveTable() {
  return (
    <table className="mt-2 w-full text-left text-sm">
      <thead className="text-fg-3">
        <tr>
          <th scope="col" className="py-2 font-medium">Epoch</th>
          {curves.map((c) => (
            <th key={c.name} scope="col" className="py-2 text-right font-medium">{c.name}</th>
          ))}
        </tr>
      </thead>
      <tbody className="divide-y divide-line">
        {curves[0].points.map((p, i) => (
          <tr key={p.epoch}>
            <th scope="row" className="num py-2 font-normal">{p.epoch}</th>
            {curves.map((c) => (
              <td key={c.name} className="num py-2 text-right">{c.points[i] ? c.points[i].value.toFixed(4) : "not run"}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>  );
}
