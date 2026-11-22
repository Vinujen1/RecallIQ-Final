import type { Domain } from "./api/types";

const DATE = new Intl.DateTimeFormat("en-CA", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });

/** ISO calendar date (YYYY-MM-DD) to "Nov 12, 2025", formatted in UTC so it never shifts a day. */
export function formatDate(iso: string | null): string | null {
  if (!iso) return null;
  const date = new Date(iso.slice(0, 10) + "T00:00:00Z");
  return Number.isNaN(date.getTime()) ? null : DATE.format(date);
}

export const DOMAIN_LABEL: Record<Domain, string> = {
  food: "Food",
  consumer_product: "Consumer product",
};

export const OFFICIAL_SITE = "https://recalls-rappels.canada.ca/en";
export const OFFICIAL_SEARCH_URL = "https://recalls-rappels.canada.ca/en/search/site";
export const COVERAGE_NOTE = "Searches a 1,000-notice sample of Government of Canada recalls, not all recalls.";

export const fixed = (value: number, digits = 3) => value.toFixed(digits);

export const signed = (value: number, digits = 3) => (value >= 0 ? "+" : "−") + Math.abs(value).toFixed(digits);
