import { expect, test } from "@playwright/test";
import { fixed, formatDate, signed } from "@/lib/format";
import { DEFAULT_MODEL, isSearchModel, MODELS } from "@/lib/models";
import { readSearchState, recallHref, searchHref, searchQuery } from "@/lib/search-params";

const params = (query: string) => new URLSearchParams(query);

test.describe("search URL state", () => {
  test("defaults: BM25, no filters, no selection", () => {
    expect(DEFAULT_MODEL).toBe("bm25");
    expect(readSearchState(params(""))).toEqual({ q: "", model: "bm25", domain: null, cat: null, id: null });
  });

  test("rejects invalid values instead of passing them to the API", () => {
    const state = readSearchState(params("q=%20cheese%20&model=gpt&domain=drugs&id=-4&cat=%20%20"));
    expect(state).toEqual({ q: "cheese", model: "bm25", domain: null, cat: null, id: null });
    expect(readSearchState(params("id=1.5")).id).toBeNull();
    expect(readSearchState(params("q=" + "x".repeat(600))).q).toHaveLength(500);
  });

  test("round-trips every field and omits defaults from links", () => {
    const state = { q: "peanut butter", model: "minilm_hard_negative" as const, domain: "food" as const, cat: "Dairy", id: 49011 };
    expect(readSearchState(params(searchQuery(state)))).toEqual(state);
    expect(searchHref({ q: "helmet", model: "bm25" })).toBe("/search?q=helmet");
    expect(searchHref({})).toBe("/search");
  });

  test("recall links remember the search they came from", () => {
    expect(recallHref(78486)).toBe("/recalls/78486");
    const href = recallHref(78486, { q: "rowing machine", model: "minilm", domain: null, cat: null, id: null });
    expect(href).toBe("/recalls/78486?q=rowing+machine&model=minilm&id=78486");
  });

  test("the four models keep their fixed public names", () => {
    expect(MODELS.map((m) => [m.key, m.name])).toEqual([
      ["bm25", "BM25"],
      ["minilm", "Model A"],
      ["minilm_finetuned", "Model B"],
      ["minilm_hard_negative", "Model C"],
    ]);
    expect(isSearchModel("minilm_finetuned")).toBe(true);
    expect(isSearchModel("model_c")).toBe(false);
  });
});

test.describe("formatting", () => {
  test("calendar dates never shift a day across time zones", () => {
    expect(formatDate("2025-11-12")).toBe("Nov 12, 2025");
    expect(formatDate("2012-01-01")).toBe("Jan 1, 2012");
    expect(formatDate(null)).toBeNull();
    expect(formatDate("not a date")).toBeNull();
  });

  test("signed differences use a true minus sign", () => {
    expect(signed(0.0026)).toBe("+0.003");
    expect(signed(-0.0033, 4)).toBe("−0.0033");
    expect(fixed(0.97024)).toBe("0.970");
  });
});
