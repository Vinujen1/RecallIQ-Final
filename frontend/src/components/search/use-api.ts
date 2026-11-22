"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { isAbort } from "@/lib/api/client";

interface Settled<T> {
  key: string;
  data?: T;
  error?: unknown;
}

/**
 * Fetches when `key` changes (null = idle), cancelling the previous request.
 * Loading is derived (latest settled key differs from the requested key), so no state is
 * set synchronously inside the effect. `previous` keeps the last data to dim, not blank, it.
 */
export function useApi<T>(key: string | null, fetcher: (signal: AbortSignal) => Promise<T>) {
  const [settled, setSettled] = useState<Settled<T> | null>(null);
  const [attempt, setAttempt] = useState(0);
  const fetcherRef = useRef(fetcher);

  useEffect(() => {
    fetcherRef.current = fetcher;
  });

  const requestKey = key === null ? null : key + "#" + attempt;

  useEffect(() => {
    if (requestKey === null) return;
    const controller = new AbortController();
    fetcherRef.current(controller.signal).then(
      (data) => setSettled({ key: requestKey, data }),
      (error) => {
        if (!isAbort(error)) setSettled({ key: requestKey, error });
      },
    );
    return () => controller.abort();
  }, [requestKey]);

  const current = settled !== null && settled.key === requestKey;
  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  return {
    data: current ? settled.data : undefined,
    error: current ? settled.error : undefined,
    loading: requestKey !== null && !current,
    previous: settled?.data,
    retry,
  };
}
