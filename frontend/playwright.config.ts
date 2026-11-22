import { defineConfig, devices } from "@playwright/test";

/**
 * Frontend regression suite.
 *   tests/unit  pure logic (URL state, formatting); no browser, no servers.
 *   tests/e2e   key user journeys against the real stack.
 *
 * The e2e tests need the FastAPI backend (http://localhost:8000, real data) and a
 * production build of this app. Start the API yourself; the frontend is started here
 * (`next start`) unless one is already running on the base URL.
 * Uses the locally installed Google Chrome (channel "chrome"), so no browser download.
 */
const baseURL = process.env.E2E_BASE_URL ?? "http://localhost:3000";

export default defineConfig({
  testDir: "./tests",
  timeout: 45_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  outputDir: "./test-results",
  use: {
    baseURL,
    channel: "chrome",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "unit", testMatch: /unit\/.*\.spec\.ts/ },
    {
      name: "desktop",
      testMatch: /e2e\/(journeys|states)\.spec\.ts/,
      use: { ...devices["Desktop Chrome"], channel: "chrome", viewport: { width: 1440, height: 900 } },
    },
    {
      name: "mobile",
      testMatch: /e2e\/mobile\.spec\.ts/,
      use: { channel: "chrome", viewport: { width: 375, height: 812 }, isMobile: true, hasTouch: true },
    },
  ],
  webServer: {
    command: "npm run start",
    url: baseURL,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
