/**
 * API client for the EcoLogic ML backend.
 *
 * Talks to a Gradio app via its auto-generated HTTP API:
 *   POST /api/{fn_name}   Body: {"data": [arg1, ...]}   Response: {"data": [result]}
 */

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:7860";

// ---------------------------------------------------------------------------
// Gradio helper
// ---------------------------------------------------------------------------

async function gradioPost<T>(fnName: string, args: unknown[]): Promise<T> {
  const response = await fetch(`${API_BASE}/api/${fnName}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: args }),
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Request to ${fnName} failed (${response.status})`);
  }
  const body = await response.json();
  // Gradio wraps results in {"data": [result]}.  Our functions return JSON
  // strings, so the first element is a string we need to parse.
  const raw: unknown = Array.isArray(body.data) ? body.data[0] : body.data;
  return typeof raw === "string" ? (JSON.parse(raw) as T) : (raw as T);
}

// ---------------------------------------------------------------------------
// Public helpers (same signatures the rest of the app already uses)
// ---------------------------------------------------------------------------

export type Dataset = {
  key: string;
  label: string;
  source: string;
};

export async function fetchDatasets(): Promise<Dataset[]> {
  return gradioPost<Dataset[]>("list_datasets", []);
}

export async function fetchAnalytics(datasetKey: string): Promise<AnalyticsPayload> {
  return gradioPost<AnalyticsPayload>("get_analytics", [datasetKey]);
}

export async function fetchPrediction(
  datasetKey: string,
  date: string,
): Promise<PredictionPayload> {
  return gradioPost<PredictionPayload>("predict", [datasetKey, date]);
}

// ---------------------------------------------------------------------------
// Types (unchanged)
// ---------------------------------------------------------------------------

export type Point = Record<string, number | string>;

export type AnalyticsPayload = {
  dataset: Dataset;
  overview: {
    dataset: string;
    timeRange: { start: string; end: string };
    observations: number;
    missingValues: number;
    duplicateTimestamps: number;
    samplingFrequency: string;
    summaryStatistics: Record<string, number>;
  };
  analysis: {
    historical: Point[];
    dailyDemand: Point[];
    rolling: Point[];
    distribution: number[];
    profiles: Record<string, Point[]>;
    calendarHeatmap: { x: number[]; y: number[]; z: number[][] };
  };
  statistics: {
    stationarity: { adfStatistic: number; pValue: number; criticalValues: Record<string, number>; interpretation: string };
    autocorrelation: { acf: Point[]; pacf: Point[] };
    correlations: { feature: string; correlation: number }[];
    seasonalityInsights: string[];
  };
  features: {
    catalog: { group: string; features: string[]; description: string }[];
    importance: { feature: string; importance: number }[];
  };
  forecasting: {
    strategy: Record<string, { start: string; end: string; rows: number }>;
    model: { bestIteration: number; bestValidationScore: number; trainingCurve: { trainMae: number[]; validMae: number[] } };
    validMetrics: Metrics;
    testMetrics: Metrics;
    baselineComparison: { model: string; mae: number }[];
    modelComparison: ModelComparisonRow[];
    baselineImprovement: number;
    walkForward?: Record<string, unknown>;
    series: { full: Point[]; zoom: Point[]; worstPredictions: Point[] };
  };
  explainability: {
    available: boolean;
    message: string;
    beeswarm: { feature: string; meanAbsShap: number }[];
    dependence: Point[];
    dependenceFeature: string | null;
    waterfall: { feature: string; featureValue: number; shapValue: number }[];
  };
  errors: {
    groups: { byHour: Point[]; byWeekday: Point[]; byMonth: Point[] };
    heatmap: { x: number[]; y: number[]; z: number[][] };
    series: { full: Point[]; zoom: Point[]; worstPredictions: Point[] };
  };
  summary: {
    observations: string[];
    strengths: string[];
    limitations: string[];
    futureImprovements: string[];
  };
};

export type Metrics = {
  mae: number;
  rmse: number;
  r2: number;
  mape: number;
  wmape: number;
  meanBias: number;
};

export type PredictionPayload = {
  dataset: string;
  date: string;
  split: string;
  averagePrediction: number;
  averageActual: number;
  points: {
    timestamp: string;
    actual: number;
    predicted: number;
    residual: number;
    absoluteError: number;
  }[];
};

export type ModelComparisonRow = {
  model: string;
  type: string;
  status: string;
  mae: number | null;
  rmse: number | null;
  r2: number | null;
  mape: number | null;
  wmape: number | null;
  meanBias: number | null;
  deltaMaeVsBest?: number | null;
};
