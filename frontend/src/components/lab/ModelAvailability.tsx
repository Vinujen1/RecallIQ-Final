"use client";

import { api } from "@/lib/api/client";
import type { SearchModel } from "@/lib/api/types";
import { useApi } from "../search/use-api";

/** Live status from GET /search/models; says so plainly when the API can't be reached. */
export function ModelAvailability({ model }: { model: SearchModel }) {
  const { data, error } = useApi("models", (signal) => api.models(signal));
  if (error) return <span className="text-sm text-fg-3">Server status unknown</span>;
  if (!data) return <span className="text-sm text-fg-3">Checking server</span>;
  const status = data.models.find((m) => m.model === model);
  return status?.available ? (
    <span className="text-sm text-fg-2">Available on this server</span>
  ) : (
    <span className="text-sm text-danger">Unavailable on this server</span>
  );
}
