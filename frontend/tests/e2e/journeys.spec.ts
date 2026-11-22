import { expect, test, type Page } from "@playwright/test";
import lab from "../../src/data/model-lab.json";

// Key user journeys against the real FastAPI backend and real 1,000-notice index.
// Assertions use records known to exist in the frozen index (see src/data/examples.ts).

const rows = (page: Page) => page.locator("ol[aria-label='Closest notices'] > li");
// The visible results label ("6 closest notices, ranked by ...") and the screen-reader live
// region that announces the same result ("6 closest notices for <query>, ranked by ...") both
// contain the ranking method, so each is located explicitly instead of by a shared substring.
const resultsLabel = (page: Page) => page.getByText(/^\d+ closest notices?, ranked by /);
const announcement = (page: Page) => page.locator("p[aria-live='polite']");

async function applyFilter(page: Page, bodyContains: string, change: () => Promise<unknown>) {
  const response = page.waitForResponse(
    (r) => r.url().endsWith("/search") && (r.request().postData() ?? "").includes(bodyContains),
  );
  await change();
  await response;
  await expect(page.locator("ol[aria-label='Closest notices']").locator("xpath=..")).toHaveAttribute("aria-busy", "false");
}

async function expectKeywordRanking(page: Page) {
  await expect(resultsLabel(page)).toBeVisible();
  await expect(resultsLabel(page)).toHaveText(/ranked by shared keywords$/);
  await expect(announcement(page)).toContainText("ranked by shared keywords");
}

test.describe("home and search", () => {
  test("hero search lands on real results with the coverage note and official link", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(/Know what you own\.\s*Find what matters\./);
    await page.getByRole("searchbox", { name: "Describe the product" }).fill("Bell youth bike helmet");
    await page.getByRole("button", { name: "Search", exact: true }).click();

    await expect(page).toHaveURL(/\/search\?q=Bell\+youth\+bike\+helmet$/);
    await expect(rows(page).first()).toContainText("Bell Cadence Youth Helmets");
    await expectKeywordRanking(page);
    await expect(page.getByText("Searches a 1,000-notice sample of Government of Canada recalls, not all recalls.")).toBeVisible();
    await expect(page.getByRole("link", { name: /Search all official recalls/ }).first()).toHaveAttribute(
      "href",
      /^https:\/\/recalls-rappels\.canada\.ca\//,
    );
  });

  test("an empty hero search explains what to type and stays put", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Search", exact: true }).click();
    await expect(page.locator("#hero-q-error")).toHaveText(/Describe a product first/);
    await expect(page).toHaveURL(/\/$/);
  });

  test("selecting a result fills the panel without adding history; model changes do", async ({ page }) => {
    await page.goto("/search?q=peanut+butter+ice+cream+salmonella");
    await expect(rows(page)).not.toHaveCount(0);
    const before = await page.evaluate(() => history.length);

    const second = rows(page).nth(1).locator("a").first();
    const title = (await second.locator("span.line-clamp-2").innerText()).trim();
    await second.click();
    await expect(page).toHaveURL(/[?&]id=\d+/);
    await expect(page.locator("#detail-title")).toHaveText(title);
    expect(await page.evaluate(() => history.length)).toBe(before);

    await page.locator("#refine-model").selectOption("minilm_hard_negative");
    await expect(page).toHaveURL(/model=minilm_hard_negative/);
    await expect(page.getByText("Semantic models always return their nearest notices")).toBeVisible();
    await page.goBack();
    await expect(page).not.toHaveURL(/model=/);
    await expectKeywordRanking(page);
  });

  test("a model number in the query is compared with the notice, both sides shown", async ({ page }) => {
    await page.goto("/search?q=NordicTrack+rowing+machine+model+NTRW19147");
    const panel = page.locator("section[aria-labelledby='detail-title']");
    await expect(panel.locator("#detail-title")).toContainText("NordicTrack NTRW19147");
    await expect(panel.getByText("You entered:")).toBeVisible();
    await expect(panel.getByText("NTRW19147", { exact: true })).toBeVisible();
    await expect(panel.getByText("written differently")).toBeVisible();
    await expect(panel.getByRole("link", { name: /Read the official notice/ })).toHaveAttribute("href", /recalls-rappels\.canada\.ca/);
  });

  test("filters are applied by the API and kept in the URL", async ({ page }) => {
    // A query with matches both inside and outside the category, so the category filter must
    // visibly narrow the results (a query with no in-category matches correctly returns none).
    await page.goto("/search?q=choking+hazard");
    await expect(rows(page)).not.toHaveCount(0);
    // Each step waits for the API response that carries the new filter and for the list to
    // stop loading, so rows are never read from the previous (dimmed) result set.
    await applyFilter(page, "consumer_product", () => page.locator("#refine-domain").selectOption("consumer_product"));
    await expect(page).toHaveURL(/domain=consumer_product/);
    const beforeCategory = await rows(page).locator("a").allInnerTexts();
    expect(beforeCategory.length).toBeGreaterThan(0);
    expect(beforeCategory.some((text) => !text.includes("Toys and games"))).toBe(true);
    await applyFilter(page, "Toys and games", () => page.locator("#refine-cat").selectOption({ label: "Toys and games (77)" }));
    await expect(page).toHaveURL(/cat=Toys\+and\+games/);
    await expect(rows(page)).not.toHaveCount(0);
    // Every row shown must belong to the chosen category.
    for (const text of await rows(page).locator("a").allInnerTexts()) expect(text).toContain("Toys and games");
  });

  test("no matching words gives an honest empty state, never a safety claim", async ({ page }) => {
    await page.goto("/search?q=helmet&cat=Sports+and+fitness");
    await expect(page.getByRole("heading", { name: "No notices in this sample match your words and filters" })).toBeVisible();
    await expect(page.getByText(/says nothing about whether the product is safe/)).toBeVisible();
    await page.getByRole("button", { name: "Remove filters" }).click();
    await expect(page).not.toHaveURL(/cat=/);
  });

  test("the evidence summary is labelled as generated and carries the disclaimer", async ({ page }) => {
    await page.goto("/search?q=peanut+butter+ice+cream+salmonella");
    await page.getByRole("button", { name: /Summarise the evidence/ }).click();
    const summary = page.locator("#evidence-summary");
    await expect(summary).toContainText("not an official statement");
    await expect(summary).toContainText("not confirmation");
  });
});

test.describe("recall pages", () => {
  test("a recall page shows the record and returns to the exact search", async ({ page }) => {
    await page.goto("/recalls/78486?q=rowing+machine&model=minilm");
    await expect(page.getByRole("heading", { level: 1 })).toContainText("NordicTrack NTRW19147");
    await expect(page.getByRole("link", { name: /Read the official notice/ })).toHaveAttribute("href", /recalls-rappels\.canada\.ca/);
    await page.getByRole("link", { name: "Back to results" }).click();
    await expect(page).toHaveURL(/\/search\?q=rowing\+machine&model=minilm/);
  });

  test("missing or malformed recall ids return a real 404 status", async ({ page, request }) => {
    for (const path of ["/recalls/1", "/recalls/abc", "/recalls/0"]) {
      expect((await request.get(path)).status(), path).toBe(404);
    }
    expect((await request.get("/recalls/78486")).status()).toBe(200);
    await page.goto("/recalls/1");
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(/isn.t in RecallIQ/);
  });
});

test.describe("model lab, method page and theme", () => {
  test("model lab shows the generated proxy numbers with their label", async ({ page }) => {
    await page.goto("/model-lab");
    await expect(page.getByText("These are proxy results, not human judgments")).toBeVisible();
    const bm25 = lab.systems.find((s) => s.key === "bm25")!;
    await expect(page.getByRole("row", { name: /^BM25/ })).toContainText(bm25.mrr10.toFixed(3));
    await expect(page.getByText("Available on this server").first()).toBeVisible();

    await page.getByRole("button", { name: "Compare the four models" }).click();
    await expect(page.locator("section[aria-labelledby^='cmp-'] ol")).toHaveCount(4);
  });

  test("the FAQ answers the safety question plainly", async ({ page }) => {
    await page.goto("/how-it-works");
    await page.getByText("Does no result mean my product is safe?").click();
    await expect(page.getByText(/^No\. RecallIQ covers a 1,000-notice sample/)).toBeVisible();
  });

  test("dark by default; a light choice persists across reloads", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await page.getByRole("radio", { name: "Light" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    await expect(page.getByRole("radio", { name: "Light" })).toHaveAttribute("aria-checked", "true");
  });
});
