import { expect, test } from "@playwright/test";

// Phone-width journeys (375 x 812, touch).

const ROUTES = [
  "/",
  "/search?q=NordicTrack+rowing+machine+model+NTRW19147",
  "/recalls/78486?q=NordicTrack+rowing+machine+model+NTRW19147",
  "/model-lab",
  "/how-it-works",
];

for (const route of ROUTES) {
  test(`no horizontal overflow: ${route}`, async ({ page }) => {
    await page.goto(route);
    await page.waitForLoadState("networkidle");
    const offenders = await page.evaluate(() => {
      const width = document.documentElement.clientWidth;
      return [...document.querySelectorAll("body *")]
        .filter((el) => {
          const box = el.getBoundingClientRect();
          // Wide tables scroll inside their own container by design.
          return box.width > 0 && box.right > width + 0.5 && !el.closest(".sr-only, .overflow-x-auto");
        })
        .slice(0, 5)
        .map((el) => el.tagName + "." + String(el.className).slice(0, 40));
    });
    expect(offenders).toEqual([]);
  });
}

test("refine sheet: opens with focus inside, Escape closes and returns focus", async ({ page }) => {
  await page.goto("/search?q=peanut+butter+ice+cream+salmonella");
  const trigger = page.getByRole("button", { name: /^Refine/ });
  await trigger.click();
  const sheet = page.getByRole("dialog", { name: "Refine search" });
  await expect(sheet).toBeVisible();
  expect(await sheet.evaluate((el) => el.contains(document.activeElement))).toBe(true);
  await page.keyboard.press("Escape");
  await expect(sheet).toBeHidden();
  await expect(trigger).toBeFocused();
});

test("tapping a result opens its page; Back to results restores the search", async ({ page }) => {
  await page.goto("/search?q=peanut+butter+ice+cream+salmonella");
  await page.locator("ol[aria-label='Closest notices'] > li a").first().click();
  await expect(page).toHaveURL(/\/recalls\/\d+\?q=peanut/);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await page.getByRole("link", { name: "Back to results" }).click();
  await expect(page).toHaveURL(/\/search\?q=peanut\+butter\+ice\+cream\+salmonella/);
});

test("mobile menu opens and navigates", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Open menu" }).click();
  await page.locator("#mobile-nav").getByRole("link", { name: "Model Lab" }).click();
  await expect(page).toHaveURL(/\/model-lab$/);
  await expect(page.locator("#mobile-nav")).toBeHidden();
});

test("search controls meet the 44px touch-target size", async ({ page }) => {
  await page.goto("/search?q=helmets");
  await page.waitForLoadState("networkidle");
  const small = await page.evaluate(() =>
    [...document.querySelectorAll("main button, main input, main select, header button, footer [role=radio]")]
      .map((el) => ({ el, box: el.getBoundingClientRect() }))
      .filter(({ box }) => box.width > 0 && box.height < 43.5)
      .map(({ el, box }) => el.tagName + ":" + (el.textContent ?? "").trim().slice(0, 24) + " " + Math.round(box.height)),
  );
  expect(small).toEqual([]);
});
