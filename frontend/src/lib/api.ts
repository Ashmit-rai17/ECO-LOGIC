export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export type Dataset = {
  key: string;
  label: string;
  source: string;
};

export async function fetchDatasets(): Promise<Dataset[]> {
  const response = await fetch(`${API_BASE}/api/datasets`, { cache: "no-store" });
  if (!response.ok) throw new Error("Unable to load datasets");
  return response.json();
}

export async function fetchAnalytics(datasetKey: string): Promise<AnalyticsPayload> {
  const response = await fetch(`${API_BASE}/api/datasets/${datasetKey}/analytics`, { cache: "no-store" });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail ?? "Unable to compute analytics");
  }
  return response.json();
}

export async function fetchPrediction(datasetKey: string, date: string): Promise<PredictionPayload> {
  const response = await fetch(`${API_BASE}/api/datasets/${datasetKey}/predict?date=${encodeURIComponent(date)}`, { cache: "no-store" });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail ?? "Unable to compute prediction");
  }
  return response.json();
}

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
