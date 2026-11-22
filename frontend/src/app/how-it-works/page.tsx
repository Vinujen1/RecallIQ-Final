// Method page: pipeline, models, identifier checks, RAG safeguards, limits and FAQ (static).
import type { Metadata } from "next";
import Link from "next/link";
import { ExternalLink } from "@/components/ui/ExternalLink";
import { VERDICT, VerdictBadge } from "@/components/ui/VerdictBadge";
import { OFFICIAL_SEARCH_URL, OFFICIAL_SITE } from "@/lib/format";

export const metadata: Metadata = {
  title: "How it works",
  description: "How RecallIQ searches recall notices, compares product codes and explains its results, and where it stops.",
};

const PIPELINE = [
  { title: "Your description", body: "Plain words, and any UPC, model, lot or best-before code you type." },
  { title: "Ranking", body: "BM25 or a MiniLM model orders the 1,000 notices by shared keywords or meaning." },
  { title: "Code check", body: "Codes in your search are compared exactly with codes stored for each notice." },
  { title: "Evidence summary", body: "On request, a short summary of the top notices; its codes, names, links and safety wording are checked first." },
  { title: "Official notice", body: "Every result links to the Government of Canada page it came from." },
];

const FAQ = [
  {
    q: "Does no result mean my product is safe?",
    a: "No. RecallIQ covers a 1,000-notice sample, and wording can differ from your description. Search the official database as well.",
  },
  {
    q: "Does a top result mean my product is recalled?",
    a: "No. It means the notice is the closest in this sample to what you typed. Compare the brand, product, sizes and codes in the official notice.",
  },
  {
    q: "Why do the meaning-based models always show results?",
    a: "The MiniLM models rank every notice by similarity and always return their nearest ones, even when none is about your product. BM25 only returns notices that share words with your search.",
  },
  {
    q: "Is a higher score a higher chance of a recall?",
    a: "No. Scores only order notices for one search with one model. They are not probabilities, and BM25 and MiniLM scores cannot be compared. RecallIQ shows ranks instead.",
  },
];

export default function HowItWorksPage() {
  return (
    <div className="mx-auto w-full max-w-[1200px] px-4 pb-24 pt-10 sm:px-6 lg:px-10 lg:pt-14">
      <header className="max-w-3xl">
        <p className="kicker">How it works</p>
        <h1 className="h1 mt-3">A faster path to the right official notice, with its limits in plain view</h1>
        <p className="mt-5 text-lg leading-relaxed text-fg-2">
          RecallIQ does one job: it finds Government of Canada recall notices that resemble a product you describe, and
          shows the evidence. Whether a recall applies to your product is decided by the official notice.
        </p>
      </header>

      <section aria-labelledby="pipeline" className="mt-16">
        <h2 id="pipeline" className="h2">From search to source</h2>
        <ol className="mt-10 flex flex-col lg:flex-row">
          {PIPELINE.map((step, index) => (
            <li key={step.title} className="flex flex-col lg:flex-1 lg:flex-row">
              <div className="flex gap-4 lg:block lg:pr-4">
                <span
                  aria-hidden
                  className={
                    "num flex h-9 w-9 shrink-0 items-center justify-center rounded-full border text-sm " +
                    (index === PIPELINE.length - 1 ? "border-accent bg-accent-fill text-on-accent" : "border-line-strong text-accent")
                  }
                >
                  {index + 1}
                </span>
                <div className="pb-2 lg:pt-4">
                  <h3 className="font-medium">{step.title}</h3>
                  <p className="mt-1.5 text-sm leading-relaxed text-fg-2">{step.body}</p>
                </div>
              </div>
              {index < PIPELINE.length - 1 && (
                <span aria-hidden className="ml-[17px] h-8 w-px bg-line-strong lg:ml-0 lg:mt-[17px] lg:h-px lg:w-8 lg:shrink-0" />
              )}
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="data" className="mt-16 grid gap-10 lg:grid-cols-[1fr_1.3fr]">
        <div>
          <h2 id="data" className="h2">The data</h2>
          <p className="mt-3 text-fg-2">
            Notices come from{" "}
            <ExternalLink href={OFFICIAL_SITE}>Recalls and safety alerts</ExternalLink>, published by the Government of Canada.
          </p>
        </div>
        <dl className="divide-y divide-line border-y border-line">
          {[
            ["Size", "1,000 notices: a fixed sample, not the full database."],
            ["Kinds", "Food recalls and consumer product recalls, including archived notices."],
            ["Fields used for search", "Title, product, issue and category text as stored from each notice."],
            ["Dates", "RecallIQ shows each notice's last-updated date. It is not always the original recall date."],
            ["Language", "English only. Official notices are also available in French."],
          ].map(([term, detail]) => (
            <div key={term} className="grid gap-1 py-4 sm:grid-cols-[12rem_1fr] sm:gap-6">
              <dt className="font-medium">{term}</dt>
              <dd className="text-fg-2">{detail}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section aria-labelledby="codes" className="mt-16 grid gap-10 lg:grid-cols-[1fr_1.3fr]">
        <div>
          <h2 id="codes" className="h2">How codes are compared</h2>
          <p className="mt-3 text-fg-2">
            When your search includes a UPC, model number, lot code or best-before date, RecallIQ compares it character for
            character with the codes stored for each notice. No model is involved, and nothing is inferred. Formats that can&apos;t
            be compared safely, such as date ranges, count as not comparable.
          </p>
        </div>
        <ul className="divide-y divide-line border-y border-line">
          {(["MATCH", "MISMATCH", "UNKNOWN"] as const).map((verdict) => (
            <li key={verdict} className="grid gap-2 py-4 sm:grid-cols-[11rem_1fr] sm:gap-6">
              <div>
                <VerdictBadge verdict={verdict} />
              </div>
              <p className="text-fg-2">{VERDICT[verdict].sentence}</p>
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="summary" className="mt-16 grid gap-10 lg:grid-cols-[1fr_1.3fr]">
        <div>
          <h2 id="summary" className="h2">What the evidence summary is</h2>
        </div>
        <div className="space-y-3 text-fg-2">
          <p>
            When AI summaries are switched on, a language model (GPT-4o-mini) writes a short summary using only the fields stored
            for up to three top notices: title, product, issue, category, recall class, date and what the notice says to do. It is
            labelled as AI-written and lists the notices it relied on, each linked to the official page.
          </p>
          <p>
            Before it is shown, RecallIQ checks the summary against those notices. Any code, number, recall ID, name or quotation
            that is not in them, any link that is not an official notice, any claim that a product is safe, or any claim that a code
            matches when the exact code check did not find a match causes it to be rejected.
          </p>
          <p>
            If there is no evidence to summarise, the summary is rejected, or the AI service is unavailable, a fixed template
            restates the stored fields instead; when the AI step failed, it says why. These checks catch invented details and
            unsafe claims, but cannot prove every reworded sentence, so always read the official notice. Notices whose codes
            conflict with yours are set aside and counted.
          </p>
        </div>
      </section>

      <section aria-labelledby="faq" className="mt-16">
        <h2 id="faq" className="h2">Questions people ask</h2>
        <div className="mt-6 divide-y divide-line border-y border-line">
          {FAQ.map((item) => (
            <details key={item.q} className="group py-1">
              <summary className="flex min-h-14 cursor-pointer list-none items-center justify-between gap-4 font-medium">
                {item.q}
                <span aria-hidden className="text-fg-3 transition-transform duration-200 group-open:rotate-45">+</span>
              </summary>
              <p className="measure pb-5 text-fg-2">{item.a}</p>
            </details>
          ))}
        </div>
      </section>

      <section className="mt-16 flex flex-col gap-4 rounded-[16px] border border-line bg-surface p-6 sm:flex-row sm:items-center sm:justify-between sm:p-8">
        <p className="max-w-xl text-fg-2">Ready to look something up? Start with the brand and what the product is.</p>
        <div className="flex flex-wrap gap-3">
          <Link href="/search" className="btn btn-primary">
            Search recall notices
          </Link>
          <ExternalLink href={OFFICIAL_SEARCH_URL} className="btn btn-secondary">
            Search all official recalls
          </ExternalLink>
        </div>
      </section>
    </div>
  );
}
