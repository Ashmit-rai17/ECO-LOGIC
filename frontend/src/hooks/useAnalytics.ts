"use client";

import { useEffect, useState } from "react";
import { AnalyticsPayload, Dataset, fetchAnalytics, fetchDatasets } from "@/lib/api";

export function useAnalytics() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selected, setSelected] = useState("AEP");
  const [analytics, setAnalytics] = useState<AnalyticsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDatasets()
      .then((items) => {
        setDatasets(items);
        if (items.length && !items.some((item) => item.key === selected)) setSelected(items[0].key);
      })
      .catch((err: Error) => setError(err.message));
  }, [selected]);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchAnalytics(selected)
      .then(setAnalytics)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [selected]);

  return { datasets, selected, setSelected, analytics, loading, error };
}
