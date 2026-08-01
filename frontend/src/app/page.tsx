"use client";

import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  BarChart3,
  Brain,
  ChartNoAxesCombined,
  Database,
  FileText,
  LineChart,
  SlidersHorizontal,
} from "lucide-react";

import { Plot } from "@/components/Plot";
import { Card, DatasetSelector, EnergyFigurine, Kpi, PageHeader, PageShell, PixelButton, Preloader, Section, SimpleTable } from "@/components/ui";
import { useAnalytics } from "@/hooks/useAnalytics";
import { fetchPrediction, PredictionPayload } from "@/lib/api";
import { formatDate, formatNumber } from "@/lib/utils";

const navItems = [
  { id: "overview", label: "Overview", icon: Database },
  { id: "time-series", label: "Time Series", icon: LineChart },
  { id: "statistics", label: "Statistics", icon: Activity },
  { id: "features", label: "Features", icon: SlidersHorizontal },
  { id: "forecasting", label: "Forecasting", icon: ChartNoAxesCombined },
  { id: "explainability", label: "Explainability", icon: Brain },
  { id: "errors", label: "Errors", icon: BarChart3 },
  { id: "summary", label: "Summary", icon: FileText },
];

const weekdayNames = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export default function Home() {
  const { datasets, selected, setSelected, analytics, loading, error } = useAnalytics();
  const [active, setActive] = useState("overview");

  const content = useMemo(() => {
    if (!analytics) return null;
    switch (active) {
      case "overview":
        return <Overview data={analytics} />;
      case "time-series":
        return <TimeSeries data={analytics} />;
      case "statistics":
        return <Statistics data={analytics} />;
      case "features":
        return <Features data={analytics} />;
      case "forecasting":
        return <Forecasting data={analytics} />;
      case "explainability":
        return <Explainability data={analytics} />;
      case "errors":
        return <ErrorAnalysis data={analytics} />;
      case "summary":
        return <ProjectSummary data={analytics} />;
      default:
        return <Overview data={analytics} />;
    }
  }, [active, analytics]);

  return (
    <main className="min-h-screen text-[#141d21]">
      <AnimatePresence>
        {loading ? <Preloader visible={loading} /> : null}
      </AnimatePresence>
      <PageShell>
      <div className="flex min-h-screen">
        <aside className="pixel-panel fixed bottom-0 left-0 top-0 z-40 hidden w-64 shrink-0 border-l-0 border-y-0 bg-white/90 p-5 md:block">
          <div className="mb-8 text-center">
            <EnergyFigurine type="leaf" className="mb-3" label="eco-t" />
            <div className="font-display text-2xl font-bold uppercase text-[#016e00]">Eco Logic</div>
            <div className="text-xs font-bold uppercase tracking-[0.18em] text-[#0061a5]">Energy Intelligence</div>
          </div>
          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <motion.button
                  key={item.id}
                  onClick={() => setActive(item.id)}
                  whileHover={{ x: 3 }}
                  whileTap={{ x: 5, y: 2 }}
                  className={`flex w-full items-center gap-3 px-3 py-2 text-left text-sm font-bold uppercase tracking-wide transition ${
                    active === item.id
                      ? "border-[3px] border-[#141d21] bg-[#00aaff] text-white shadow-[4px_4px_0_rgba(20,29,33,0.24)]"
                      : "border-[3px] border-transparent text-[#3e4851] hover:border-[#141d21] hover:bg-[#f4faff] hover:text-[#016e00]"
                  }`}
                >
                  <Icon size={16} />
                  {item.label}
                </motion.button>
              );
            })}
          </nav>
        </aside>

        <section className="flex-1 md:ml-64">
          <header className="sticky top-0 z-30 px-4 py-4 backdrop-blur-sm md:px-8">
            <PageHeader title="Forecasting Lab" subtitle="Notebook workflow refactored into a FastAPI + Next.js energy analytics cockpit.">
              <DatasetSelector value={selected} datasets={datasets} onChange={setSelected} />
            </PageHeader>
            <div className="mt-4 flex gap-2 overflow-x-auto md:hidden">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setActive(item.id)}
                  className={`whitespace-nowrap border-[3px] border-[#141d21] px-3 py-2 text-xs font-bold uppercase ${
                    active === item.id ? "bg-[#00aaff] text-white" : "bg-white text-[#3e4851]"
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </header>

          <div className="relative p-4 md:p-8">
            <EnergyFigurine type="wind" className="absolute right-8 top-1 hidden opacity-70 lg:flex" label="wind" />
            {error ? <ErrorState message={error} /> : null}
            <AnimatePresence mode="wait">
              {!loading && !error ? (
                <motion.div
                  key={`${active}-${selected}`}
                  initial={{ opacity: 0, y: 18 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -12 }}
                  transition={{ duration: 0.28, ease: "easeOut" }}
                >
                  {content}
                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>
        </section>
      </div>
      </PageShell>
    </main>
  );
}

function LoadingState() {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      {Array.from({ length: 6 }).map((_, index) => (
        <div key={index} className="pixel-panel h-32 animate-pulse bg-white/70" />
      ))}
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <Card>
      <div className="font-display text-xl font-bold uppercase text-[#ba1a1a]">Unable to load analytics</div>
      <p className="mt-2 text-sm text-[#3e4851]">{message}</p>
    </Card>
  );
}

function Overview({ data }: { data: NonNullable<ReturnType<typeof useAnalytics>["analytics"]> }) {
  const stats = data.overview.summaryStatistics;
  return (
    <Section title="Overview" description="Basic dataset quality and demand summary.">
      <div className="grid gap-4 md:grid-cols-3">
        <Kpi label="Dataset" value={data.dataset.key} />
        <Kpi label="Observations" value={formatNumber(data.overview.observations, 0)} />
        <Kpi label="Frequency" value={data.overview.samplingFrequency} />
        <Kpi label="Start" value={formatDate(data.overview.timeRange.start)} />
        <Kpi label="End" value={formatDate(data.overview.timeRange.end)} />
        <Kpi label="Missing Values" value={formatNumber(data.overview.missingValues, 0)} />
        <Kpi label="Duplicate Timestamps" value={formatNumber(data.overview.duplicateTimestamps, 0)} />
        <Kpi label="Mean Demand" value={`${formatNumber(stats.mean)} MW`} />
        <Kpi label="Max Demand" value={`${formatNumber(stats.max)} MW`} />
      </div>
      <SimpleTable
        columns={["Metric", "Value"]}
        rows={Object.entries(stats).map(([Metric, value]) => ({ Metric, Value: formatNumber(value) }))}
      />
    </Section>
  );
}

function TimeSeries({ data }: { data: NonNullable<ReturnType<typeof useAnalytics>["analytics"]> }) {
  return (
    <div className="space-y-8">
      <Section title="Time Series Analysis" description="Historical demand, trend, seasonality, rolling behavior, and demand distributions.">
        <Card className="relative">
          <EnergyFigurine type="hydro" className="absolute right-5 top-5 opacity-80" label="flow" />
          <Plot
            data={[{ x: data.analysis.historical.map((p) => p.timestamp), y: data.analysis.historical.map((p) => p.demand_mw), type: "scatter", mode: "lines", name: "Demand" }]}
            layout={{ title: "Historical Demand" }}
          />
        </Card>
        <div className="grid gap-4 lg:grid-cols-2">
          <Card className="relative">
            <EnergyFigurine type="solar" className="absolute right-5 top-5 opacity-80" label="solar" />
            <Plot
              data={[
                { x: data.analysis.rolling.map((p) => p.timestamp), y: data.analysis.rolling.map((p) => p.demand_mw), type: "scatter", mode: "lines", name: "Demand" },
                { x: data.analysis.rolling.map((p) => p.timestamp), y: data.analysis.rolling.map((p) => p.rolling_mean), type: "scatter", mode: "lines", name: "Rolling Mean" },
                { x: data.analysis.rolling.map((p) => p.timestamp), y: data.analysis.rolling.map((p) => p.rolling_std), type: "scatter", mode: "lines", name: "Rolling Std" },
              ]}
              layout={{ title: "Rolling Mean and Std" }}
            />
          </Card>
          <Card>
            <Plot data={[{ x: data.analysis.distribution, type: "histogram", name: "Demand" }]} layout={{ title: "Distribution" }} />
          </Card>
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          <ProfilePlot title="Hourly Pattern" xKey="hour" series={data.analysis.profiles.hourly} />
          <ProfilePlot title="Monthly Pattern" xKey="month" series={data.analysis.profiles.monthly} />
          <ProfilePlot title="Yearly Pattern" xKey="year" series={data.analysis.profiles.yearly} />
          <Card>
            <Plot
              data={[{ x: data.analysis.calendarHeatmap.x, y: data.analysis.calendarHeatmap.y.map((d) => weekdayNames[d]), z: data.analysis.calendarHeatmap.z, type: "heatmap", colorscale: "Viridis" }]}
              layout={{ title: "Hour x Weekday Heatmap" }}
            />
          </Card>
        </div>
      </Section>
    </div>
  );
}

function ProfilePlot({ title, xKey, series }: { title: string; xKey: string; series: Record<string, string | number>[] }) {
  return (
    <Card>
      <Plot
        data={[{ x: series.map((p) => p[xKey]), y: series.map((p) => p.demand_mw), type: "scatter", mode: "lines+markers", name: title }]}
        layout={{ title }}
      />
    </Card>
  );
}

function Statistics({ data }: { data: NonNullable<ReturnType<typeof useAnalytics>["analytics"]> }) {
  const adf = data.statistics.stationarity;
  return (
    <Section title="Statistical Analysis" description="Stationarity, autocorrelation, feature correlation, and seasonality notes.">
      <div className="grid gap-4 md:grid-cols-3">
        <Kpi label="ADF Statistic" value={formatNumber(adf.adfStatistic, 4)} />
        <Kpi label="p-value" value={formatNumber(adf.pValue, 4)} />
        <Kpi label="Interpretation" value={adf.pValue < 0.05 ? "Stationary" : "Non-stationary"} hint={adf.interpretation} />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <Plot
            data={[
              { x: data.statistics.autocorrelation.acf.map((p) => p.lag), y: data.statistics.autocorrelation.acf.map((p) => p.value), type: "bar", name: "ACF" },
            ]}
            layout={{ title: "Autocorrelation" }}
          />
        </Card>
        <Card>
          <Plot
            data={[
              { x: data.statistics.autocorrelation.pacf.map((p) => p.lag), y: data.statistics.autocorrelation.pacf.map((p) => p.value), type: "bar", name: "PACF" },
            ]}
            layout={{ title: "Partial Autocorrelation" }}
          />
        </Card>
      </div>
      <SimpleTable
        columns={["Feature", "Correlation"]}
        rows={data.statistics.correlations.map((row) => ({ Feature: row.feature, Correlation: formatNumber(row.correlation, 4) }))}
      />
      <InsightList items={data.statistics.seasonalityInsights} />
    </Section>
  );
}

function Features({ data }: { data: NonNullable<ReturnType<typeof useAnalytics>["analytics"]> }) {
  return (
    <Section title="Feature Engineering" description="The transformed inputs preserved from the notebook pipeline.">
      <div className="grid gap-4 md:grid-cols-2">
        {data.features.catalog.map((group) => (
          <Card key={group.group}>
            <h3 className="font-display text-xl font-bold uppercase text-[#016e00]">{group.group}</h3>
            <p className="mt-2 text-sm leading-6 text-[#3e4851]">{group.description}</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {group.features.map((feature) => (
                <span key={feature} className="border-2 border-[#141d21] bg-[#f4faff] px-2 py-1 text-xs font-bold text-[#016e00]">
                  {feature}
                </span>
              ))}
            </div>
          </Card>
        ))}
      </div>
      <Card>
        <Plot
          data={[{ x: data.features.importance.slice(0, 15).map((p) => p.importance), y: data.features.importance.slice(0, 15).map((p) => p.feature), type: "bar", orientation: "h" }]}
          layout={{ title: "Top Feature Importance", yaxis: { automargin: true } }}
        />
      </Card>
    </Section>
  );
}

function Forecasting({ data }: { data: NonNullable<ReturnType<typeof useAnalytics>["analytics"]> }) {
  const metrics = data.forecasting.testMetrics;
  const [date, setDate] = useState(data.forecasting.strategy.test.start.slice(0, 10));
  const [prediction, setPrediction] = useState<PredictionPayload | null>(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  async function handlePredict(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPredictionLoading(true);
    setPredictionError(null);
    try {
      setPrediction(await fetchPrediction(data.dataset.key, date));
    } catch (err) {
      setPredictionError(err instanceof Error ? err.message : "Unable to compute prediction");
      setPrediction(null);
    } finally {
      setPredictionLoading(false);
    }
  }

  return (
    <Section title="Forecasting" description="XGBoost forecast with the notebook train, validation, and test strategy.">
      <Card>
        <form onSubmit={handlePredict} className="flex flex-col gap-3 md:flex-row md:items-end">
          <div className="flex-1">
            <label className="text-sm font-bold uppercase tracking-wide text-[#3e4851]" htmlFor="prediction-date">
              Predict demand for a date
            </label>
            <input
              id="prediction-date"
              type="date"
              value={date}
              onChange={(event) => setDate(event.target.value)}
              className="mt-2 w-full border-[3px] border-[#141d21] bg-white px-3 py-2 text-sm font-bold text-[#016e00] shadow-[4px_4px_0_rgba(20,29,33,0.2)] outline-none focus:ring-4 focus:ring-[#00aaff]/30"
            />
            <p className="mt-2 text-xs text-[#3e4851]">
              Uses the same trained model and feature pipeline. Historical actuals are shown when the selected date exists in the dataset.
            </p>
          </div>
          <PixelButton type="submit" disabled={predictionLoading}>
            {predictionLoading ? "Predicting..." : "Predict"}
          </PixelButton>
        </form>
        {predictionError ? <p className="mt-3 text-sm text-red-300">{predictionError}</p> : null}
        {prediction ? (
          <div className="mt-5 space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              <Kpi label="Prediction Date" value={formatDate(prediction.date)} hint={`Split: ${prediction.split}`} />
              <Kpi label="Avg Predicted Demand" value={`${formatNumber(prediction.averagePrediction)} MW`} />
              <Kpi label="Avg Actual Demand" value={`${formatNumber(prediction.averageActual)} MW`} />
            </div>
            <Plot
              data={[
                { x: prediction.points.map((p) => p.timestamp), y: prediction.points.map((p) => p.actual), type: "scatter", mode: "lines+markers", name: "Actual" },
                { x: prediction.points.map((p) => p.timestamp), y: prediction.points.map((p) => p.predicted), type: "scatter", mode: "lines+markers", name: "Predicted" },
              ]}
              layout={{ title: "Hourly Prediction for Selected Date" }}
              height={320}
            />
            <SimpleTable
              columns={["Time", "Actual", "Predicted", "Absolute Error"]}
              rows={prediction.points.map((point) => ({
                Time: new Date(point.timestamp).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
                Actual: `${formatNumber(point.actual)} MW`,
                Predicted: `${formatNumber(point.predicted)} MW`,
                "Absolute Error": `${formatNumber(point.absoluteError)} MW`,
              }))}
            />
          </div>
        ) : null}
      </Card>
      <div className="grid gap-4 md:grid-cols-4">
        <Kpi label="MAE" value={`${formatNumber(metrics.mae)} MW`} />
        <Kpi label="RMSE" value={`${formatNumber(metrics.rmse)} MW`} />
        <Kpi label="R2" value={formatNumber(metrics.r2, 4)} />
        <Kpi label="WMAPE" value={`${formatNumber(metrics.wmape)}%`} />
      </div>
      <Card>
        <h3 className="mb-3 font-semibold">Model Comparison</h3>
        <Plot
          data={[
            {
              x: data.forecasting.modelComparison.filter((row) => row.mae !== null).map((row) => row.model),
              y: data.forecasting.modelComparison.filter((row) => row.mae !== null).map((row) => row.mae),
              type: "bar",
              name: "MAE",
            },
          ]}
          layout={{ title: "Test MAE by Model" }}
          height={300}
        />
        <SimpleTable
          columns={["Model", "Type", "Status", "MAE", "RMSE", "R2", "WMAPE", "Delta vs Best MAE"]}
          rows={data.forecasting.modelComparison.map((row) => ({
            Model: row.model,
            Type: row.type,
            Status: row.status,
            MAE: row.mae === null ? "n/a" : `${formatNumber(row.mae)} MW`,
            RMSE: row.rmse === null ? "n/a" : `${formatNumber(row.rmse)} MW`,
            R2: row.r2 === null ? "n/a" : formatNumber(row.r2, 4),
            WMAPE: row.wmape === null ? "n/a" : `${formatNumber(row.wmape)}%`,
            "Delta vs Best MAE": row.deltaMaeVsBest === null || row.deltaMaeVsBest === undefined ? "n/a" : `${formatNumber(row.deltaMaeVsBest)} MW`,
          }))}
        />
        <p className="mt-3 text-xs text-[#3e4851]">
          Logistic regression is shown as not applicable because the target is continuous electricity demand, not a classification label.
        </p>
      </Card>
      <SimpleTable
        columns={["Split", "Start", "End", "Rows"]}
        rows={Object.entries(data.forecasting.strategy).map(([Split, value]) => ({
          Split,
          Start: formatDate(value.start),
          End: formatDate(value.end),
          Rows: formatNumber(value.rows, 0),
        }))}
      />
      <Card className="relative">
        <EnergyFigurine type="operator" className="absolute right-5 top-5 opacity-80" label="model" />
        <Plot
          data={[
            { x: data.forecasting.series.full.map((p) => p.timestamp), y: data.forecasting.series.full.map((p) => p.actual), type: "scatter", mode: "lines", name: "Actual" },
            { x: data.forecasting.series.full.map((p) => p.timestamp), y: data.forecasting.series.full.map((p) => p.predicted), type: "scatter", mode: "lines", name: "Predicted" },
          ]}
          layout={{ title: "Actual vs Predicted" }}
        />
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <Plot
            data={[{ x: data.forecasting.baselineComparison.map((p) => p.model), y: data.forecasting.baselineComparison.map((p) => p.mae), type: "bar" }]}
            layout={{ title: "Baseline Comparison" }}
          />
        </Card>
        <Card>
          <Plot
            data={[
              { y: data.forecasting.model.trainingCurve.trainMae, type: "scatter", mode: "lines", name: "Train MAE" },
              { y: data.forecasting.model.trainingCurve.validMae, type: "scatter", mode: "lines", name: "Validation MAE" },
            ]}
            layout={{ title: "Training Curve" }}
          />
        </Card>
      </div>
    </Section>
  );
}

function Explainability({ data }: { data: NonNullable<ReturnType<typeof useAnalytics>["analytics"]> }) {
  const shap = data.explainability;
  return (
    <Section title="Explainability" description="Model interpretation through feature importance and SHAP when available.">
      <Card>
        <p className="text-sm text-[#3e4851]">{shap.message}</p>
      </Card>
      {shap.available ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <Plot data={[{ x: shap.beeswarm.map((p) => p.meanAbsShap), y: shap.beeswarm.map((p) => p.feature), type: "bar", orientation: "h" }]} layout={{ title: "SHAP Summary" }} />
          </Card>
          <Card>
            <Plot
              data={[{ x: shap.dependence.map((p) => p.featureValue), y: shap.dependence.map((p) => p.shapValue), type: "scatter", mode: "markers" }]}
              layout={{ title: `SHAP Dependence: ${shap.dependenceFeature ?? ""}` }}
            />
          </Card>
        </div>
      ) : null}
      <SimpleTable
        columns={["Feature", "Importance"]}
        rows={data.features.importance.slice(0, 15).map((row) => ({ Feature: row.feature, Importance: formatNumber(row.importance, 4) }))}
      />
    </Section>
  );
}

function ErrorAnalysis({ data }: { data: NonNullable<ReturnType<typeof useAnalytics>["analytics"]> }) {
  return (
    <Section title="Error Analysis" description="Residual behavior and where the forecast struggles.">
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <Plot data={[{ x: data.errors.series.full.map((p) => p.residual), type: "histogram" }]} layout={{ title: "Residual Distribution" }} />
        </Card>
        <Card>
          <Plot data={[{ x: data.errors.series.full.map((p) => p.timestamp), y: data.errors.series.full.map((p) => p.residual), type: "scatter", mode: "lines" }]} layout={{ title: "Residual Timeline" }} />
        </Card>
        <Card>
          <Plot data={[{ x: data.errors.groups.byHour.map((p) => p.hour), y: data.errors.groups.byHour.map((p) => p.mae), type: "scatter", mode: "lines+markers" }]} layout={{ title: "Error by Hour" }} />
        </Card>
        <Card>
          <Plot data={[{ x: data.errors.groups.byMonth.map((p) => p.month), y: data.errors.groups.byMonth.map((p) => p.mae), type: "bar" }]} layout={{ title: "Error by Month" }} />
        </Card>
      </div>
      <Card>
        <Plot data={[{ x: data.errors.heatmap.x, y: data.errors.heatmap.y.map((d) => weekdayNames[d]), z: data.errors.heatmap.z, type: "heatmap", colorscale: "Reds" }]} layout={{ title: "Forecast Error Heatmap" }} />
      </Card>
      <SimpleTable
        columns={["timestamp", "actual", "predicted", "residual", "absolute_error"]}
        rows={data.errors.series.worstPredictions.map((row) => ({
          timestamp: String(row.timestamp),
          actual: formatNumber(Number(row.actual)),
          predicted: formatNumber(Number(row.predicted)),
          residual: formatNumber(Number(row.residual)),
          absolute_error: formatNumber(Number(row.absolute_error)),
        }))}
      />
    </Section>
  );
}

function ProjectSummary({ data }: { data: NonNullable<ReturnType<typeof useAnalytics>["analytics"]> }) {
  return (
    <Section title="Project Summary" description="Portfolio-ready interpretation of the model and analysis.">
      <div className="grid gap-4 md:grid-cols-2">
        <SummaryCard title="Key observations" items={data.summary.observations} />
        <SummaryCard title="Model strengths" items={data.summary.strengths} />
        <SummaryCard title="Model limitations" items={data.summary.limitations} />
        <SummaryCard title="Future improvements" items={data.summary.futureImprovements} />
      </div>
    </Section>
  );
}

function SummaryCard({ title, items }: { title: string; items: string[] }) {
  return (
    <Card>
      <h3 className="font-display text-xl font-bold uppercase text-[#016e00]">{title}</h3>
      <InsightList items={items} />
    </Card>
  );
}

function InsightList({ items }: { items: string[] }) {
  return (
    <ul className="mt-3 space-y-2 text-sm leading-6 text-[#3e4851]">
      {items.map((item) => (
        <li key={item}>- {item}</li>
      ))}
    </ul>
  );
}
