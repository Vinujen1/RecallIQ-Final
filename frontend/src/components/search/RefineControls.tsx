"use client";

import type { CategoryCount, Domain, ModelStatus, SearchModel } from "@/lib/api/types";
import { MODELS } from "@/lib/models";

export interface RefineValue {
  model: SearchModel;
  domain: Domain | null;
  cat: string | null;
}

/** Category options for the chosen domain; counts are over the whole 1,000-notice index. */
export function categoryOptions(items: CategoryCount[] | undefined, domain: Domain | null) {
  const counts = new Map<string, number>();
  for (const item of items ?? []) {
    if (domain && item.domain !== domain) continue;
    counts.set(item.category, (counts.get(item.category) ?? 0) + item.count);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

/** Retrieval model, type and category: inline on wide screens, inside the sheet on phones. */
export function RefineControls({
  idPrefix,
  value,
  onChange,
  models,
  categories,
  stacked = false,
}: {
  idPrefix: string;
  value: RefineValue;
  onChange: (next: RefineValue) => void;
  models: ModelStatus[] | undefined;
  categories: CategoryCount[] | undefined;
  stacked?: boolean;
}) {
  const availability = new Map(models?.map((m) => [m.model, m]));
  const options = categoryOptions(categories, value.domain);
  const field = stacked ? "space-y-1.5" : "min-w-0 flex-1 space-y-1";

  return (
    <div className={stacked ? "space-y-5" : "flex flex-wrap items-end gap-3"}>
      <div className={field}>
        <label htmlFor={idPrefix + "-model"} className="text-[0.8125rem] font-medium text-fg-2">
          Retrieval model
        </label>
        <select
          id={idPrefix + "-model"}
          className="field"
          value={value.model}
          onChange={(event) => onChange({ ...value, model: event.target.value as SearchModel })}
        >
          {MODELS.map((m) => {
            const status = availability.get(m.key);
            const unavailable = status ? !status.available : false;
            return (
              <option key={m.key} value={m.key} disabled={unavailable && m.key !== value.model}>
                {m.name}: {m.short}
                {unavailable ? " (unavailable on this server)" : ""}
              </option>
            );
          })}
        </select>
      </div>
      <div className={field}>
        <label htmlFor={idPrefix + "-domain"} className="text-[0.8125rem] font-medium text-fg-2">
          Type
        </label>
        <select
          id={idPrefix + "-domain"}
          className="field"
          value={value.domain ?? ""}
          onChange={(event) => onChange({ ...value, domain: (event.target.value || null) as Domain | null, cat: null })}
        >
          <option value="">Food and consumer products</option>
          <option value="food">Food</option>
          <option value="consumer_product">Consumer products</option>
        </select>
      </div>
      <div className={field}>
        <label htmlFor={idPrefix + "-cat"} className="text-[0.8125rem] font-medium text-fg-2">
          Category
        </label>
        <select
          id={idPrefix + "-cat"}
          className="field"
          value={value.cat ?? ""}
          disabled={!categories}
          onChange={(event) => onChange({ ...value, cat: event.target.value || null })}
        >
          <option value="">{categories ? "All categories" : "Loading categories"}</option>
          {value.cat && !options.some(([name]) => name === value.cat) && <option value={value.cat}>{value.cat}</option>}
          {options.map(([name, count]) => (
            <option key={name} value={name}>
              {name} ({count})
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
