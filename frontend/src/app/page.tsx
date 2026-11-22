// Home: hero search plus a plain statement of what RecallIQ can and cannot tell you.
import Link from "next/link";
import { LabelScanVisual } from "@/components/brand/LabelScanVisual";
import { HeroSearch } from "@/components/search/HeroSearch";
import { ExternalLink } from "@/components/ui/ExternalLink";
import { VerdictBadge } from "@/components/ui/VerdictBadge";
import { OFFICIAL_SEARCH_URL, OFFICIAL_SITE } from "@/lib/format";

const STEPS = [
  {
    title: "Describe what you have",
    body: "Type it the way you would say it: the brand, what it is, what worries you. Add the UPC, model or lot code from the package if you have it.",
    note: "Plain words or codes",
    sample: <code className="num text-sm text-fg-2">peanut butter ice cream salmonella</code>,
  },
  {
    title: "See the closest official notices",
    body: "RecallIQ ranks Government of Canada recall notices by shared keywords or by meaning. Ranked notices are candidates to read, not confirmation.",
    note: "Four retrieval models, BM25 by default",
    sample: (
      <ol className="space-y-1.5 text-sm">
        <li className="flex gap-3"><span className="num text-fg-3">1</span><span className="text-fg">Closest notice</span></li>
        <li className="flex gap-3"><span className="num text-fg-3">2</span><span className="text-fg-2">Next closest</span></li>
        <li className="flex gap-3"><span className="num text-fg-3">3</span><span className="text-fg-3">And so on</span></li>
      </ol>
    ),
  },
  {
    title: "Check your codes, then read the notice",
    body: "When your search includes a code, RecallIQ compares it exactly with the codes in each notice and shows both sides. The official notice always has the final word.",
    note: "Exact comparison, no guessing",
    sample: (
      <div className="flex flex-wrap gap-2">
        <VerdictBadge verdict="MATCH" />
        <VerdictBadge verdict="MISMATCH" />
        <VerdictBadge verdict="UNKNOWN" />
      </div>
    ),
  },
];

const LIMITS = [
  ["Where the data comes from", "Recall notices published by the Government of Canada (recalls-rappels.canada.ca), for food and consumer products."],
  ["What is covered", "A 1,000-notice sample, including archived notices. Many recalls are not in this sample, so also search the official site."],
  ["What a result means", "A ranked notice is a candidate to read, not confirmation that your product is recalled."],
  ["What no result means", "Nothing about safety. It only means no notice in this sample was close to your description."],
  ["Language", "English only. Official notices are also available in French."],
];

export default function Home() {
  return (
    <>
      <section className="relative overflow-hidden">
        <div className="mx-auto grid max-w-[1360px] grid-cols-[minmax(0,1fr)] items-center gap-12 px-4 pb-16 pt-12 sm:px-6 md:pt-16 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] lg:gap-8 lg:px-10 lg:pb-24 lg:pt-20">
          <div className="min-w-0 max-w-[660px]">
            <p className="kicker rise">Canadian product recalls, searched in plain language</p>
            <h1 className="display rise mt-5 [animation-delay:80ms]">
              <span className="block sm:whitespace-nowrap">Know what you own.</span>
              <span className="block sm:whitespace-nowrap">Find what matters.</span>
            </h1>
            <p className="rise mt-6 max-w-[34rem] text-lg leading-relaxed text-fg-2 [animation-delay:160ms]">
              Explore Canadian product recalls using everyday descriptions, with evidence from official
              government notices.
            </p>
            <div className="rise mt-9 [animation-delay:240ms]">
              <HeroSearch />
            </div>
            <p className="rise mt-6 text-sm text-fg-3 [animation-delay:320ms]">
              Searches a 1,000-notice sample.{" "}
              <ExternalLink href={OFFICIAL_SEARCH_URL}>Search all official recalls</ExternalLink>
            </p>
          </div>
          <div className="relative lg:-mr-6">
            <LabelScanVisual />
          </div>
        </div>
      </section>

      <section aria-labelledby="how" className="border-t border-line">
        <div className="mx-auto max-w-[1360px] px-4 py-16 sm:px-6 md:py-24 lg:px-10">
          <div className="flex flex-col justify-between gap-6 md:flex-row md:items-end">
            <h2 id="how" className="h2 max-w-xl">
              From a description on your counter to the notice that covers it
            </h2>
            <Link href="/how-it-works" className="link shrink-0 text-[0.9375rem]">
              Read how RecallIQ works
            </Link>
          </div>
          <ol className="mt-12 grid gap-10 md:grid-cols-[1fr_1fr_1.25fr] md:gap-0 md:divide-x md:divide-line">
            {STEPS.map((step, index) => (
              <li key={step.title} className="flex flex-col border-t border-line pt-6 md:border-t-0 md:px-8 md:pt-0 md:first:pl-0 md:last:pr-0">
                <span className="num text-sm text-accent">Step {index + 1}</span>
                <h3 className="h3 mt-3">{step.title}</h3>
                <p className="mt-3 flex-1 text-[0.9375rem] leading-relaxed text-fg-2">{step.body}</p>
                <div className="mt-6 rounded-[10px] border border-line bg-surface p-4">{step.sample}</div>
                <p className="mt-3 text-[0.8125rem] text-fg-3">{step.note}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section aria-labelledby="trust" className="border-t border-line bg-surface/40">
        <div className="mx-auto grid max-w-[1360px] gap-10 px-4 py-16 sm:px-6 md:py-24 lg:grid-cols-[1fr_1.2fr] lg:px-10">
          <div>
            <h2 id="trust" className="h2">Official sources, stated limits</h2>
            <p className="mt-4 max-w-md text-fg-2">
              Every result links to the Government of Canada notice it came from. RecallIQ helps you find
              the right notice faster; it does not replace it.
            </p>
            <p className="mt-6">
              <ExternalLink href={OFFICIAL_SITE} className="btn btn-secondary">
                Visit Recalls and safety alerts
              </ExternalLink>
            </p>
          </div>
          <dl className="divide-y divide-line border-y border-line">
            {LIMITS.map(([term, detail]) => (
              <div key={term} className="grid gap-1 py-5 sm:grid-cols-[13rem_1fr] sm:gap-6">
                <dt className="font-medium text-fg">{term}</dt>
                <dd className="text-fg-2">{detail}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      <section aria-labelledby="lab" className="border-t border-line">
        <div className="mx-auto flex max-w-[1360px] flex-col gap-6 px-4 py-14 sm:px-6 md:flex-row md:items-center md:justify-between lg:px-10">
          <div className="max-w-2xl">
            <h2 id="lab" className="h3">Inside the search: four retrieval approaches</h2>
            <p className="mt-2 text-fg-2">
              A keyword baseline (BM25) and three MiniLM language models, one pretrained and two fine-tuned on
              recall notices. See how they were trained and how they compare, including what the evaluation
              cannot yet show.
            </p>
          </div>
          <Link href="/model-lab" className="btn btn-secondary shrink-0">
            Open the Model Lab
          </Link>
        </div>
      </section>
    </>
  );
}
