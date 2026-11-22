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
