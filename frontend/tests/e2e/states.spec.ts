import { expect, test } from "@playwright/test";

// Failure states, simulated by intercepting the browser's API calls.

test("API unreachable: clear message and a working retry", async ({ page }) => {
  let fail = true;
  await page.route("**/search", (route) => (fail ? route.abort() : route.continue()));
  await page.goto("/search?q=cheese");
  const alert = page.getByRole("alert").filter({ hasText: "The search didn't complete" });
  await expect(alert).toContainText("Can't reach the RecallIQ API");
  fail = false;
  await alert.getByRole("button", { name: /Try again/ }).click();
  await expect(page.locator("ol[aria-label='Closest notices'] > li")).not.toHaveCount(0);
});

test("model unavailable (503): no server details leak, BM25 offered", async ({ page }) => {
  await page.route("**/search", (route) =>
    route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "C:\\models\\secret\\best missing" }) }),
  );
  await page.goto("/search?q=cheese&model=minilm");
  const alert = page.getByRole("alert").filter({ hasText: "couldn't run" });
  await expect(alert).toContainText("unavailable on this server");
  await expect(page.getByText("secret")).toHaveCount(0);
  await expect(alert.getByRole("button", { name: "Search with BM25" })).toBeVisible();
});

test("a model the server reports unavailable is labelled and cannot be newly chosen", async ({ page }) => {
  await page.route("**/search/models", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    for (const m of body.models) if (m.model === "minilm_finetuned") Object.assign(m, { available: false, reason: "x" });
    await route.fulfill({ response, json: body });
  });
  await page.goto("/search?q=cheese");
  const option = page.locator("#refine-model").locator("option[value='minilm_finetuned']");
  await expect(option).toHaveText(/unavailable on this server/);
  await expect(option).toBeDisabled();
});

// --- POST /explain (AI summary) states: real retrieval, explanation rewritten in the browser ---

async function openSummary(page: import("@playwright/test").Page) {
  await page.goto("/search?q=peanut+butter+salmonella");
  await page.getByRole("button", { name: /Summarise the evidence/ }).click();
  return page.locator("#evidence-summary");
}

test("an AI summary is labelled, cites official notices and shows usage", async ({ page }) => {
  await page.route("**/explain", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    const first = body.retrieved_facts[0];
    body.explanation = {
      generator: "llm:openai",
      text: `Recall ${first.recall_id} may be relevant. Check the official notice.`,
      cited_recall_ids: [first.recall_id],
      fallback_reason: null,
      evidence_sufficient: true,
      generation: { provider: "openai", model: "gpt-4o-mini", outcome: "accepted", prompt_version: "rag-v2",
                    latency_ms: 1840, prompt_tokens: 800, completion_tokens: 100, total_tokens: 900 },
    };
    await route.fulfill({ response, json: body });
  });
  const summary = await openSummary(page);
  await expect(summary).toContainText("AI-written summary (gpt-4o-mini)");
  await expect(summary).toContainText("not an official statement");
  await expect(summary).toContainText("Cited in the summary");
  await expect(summary).toContainText("1.8 s · 900 tokens");
  await expect(summary).toContainText("not confirmation");
  await expect(summary.getByRole("link", { name: /Official notice for/ }).first()).toHaveAttribute("href", /recalls-rappels\.canada\.ca/);
});

test("an AI fallback says why the fixed template is shown", async ({ page }) => {
  await page.route("**/explain", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.explanation.fallback_reason = "The AI service took too long to respond; the fixed template answer is shown instead.";
    await route.fulfill({ response, json: body });
  });
  const summary = await openSummary(page);
  await expect(summary).toContainText("fixed template, no AI writing");
  await expect(summary.getByRole("status").filter({ hasText: "took too long" })).toBeVisible();
});

test("a failed summary request shows an error with a working retry", async ({ page }) => {
  let fail = true;
  await page.route("**/explain", (route) =>
    fail ? route.fulfill({ status: 500, contentType: "application/json", body: "{}" }) : route.continue(),
  );
  const summary = await openSummary(page);
  const alert = summary.getByRole("alert");
  await expect(alert).toContainText("Couldn't build the summary");
  fail = false;
  await alert.getByRole("button", { name: "Try again" }).click();
  await expect(summary).toContainText("not confirmation");
});
