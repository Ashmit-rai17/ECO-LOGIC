"use client";

import dynamic from "next/dynamic";
import type { Layout, PlotData } from "plotly.js";
import { motion } from "framer-motion";

const Plotly = dynamic(() => import("react-plotly.js"), { ssr: false });

type PlotProps = {
  data: Partial<PlotData>[];
  layout?: Partial<Layout>;
  height?: number;
};

export function Plot({ data, layout, height = 360 }: PlotProps) {
  return (
    <motion.div
      className="pixel-grid w-full overflow-hidden border-2 border-[#141d21] bg-[#f8fdff]"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      <Plotly
        data={data.map((trace, index) => ({
          marker: { color: ["#016e00", "#00aaff", "#f1c100", "#75fd43"][index % 4] },
          line: { color: ["#016e00", "#00aaff", "#f1c100", "#75fd43"][index % 4], width: 3, shape: "spline" },
          ...trace,
        }))}
        layout={{
          autosize: true,
          height,
          margin: { l: 54, r: 24, t: 50, b: 52 },
          paper_bgcolor: "rgba(255,255,255,0)",
          plot_bgcolor: "rgba(255,255,255,0.55)",
          font: { color: "#141d21", family: "Quicksand, sans-serif" },
          legend: { orientation: "h", y: -0.2 },
          transition: { duration: 650, easing: "cubic-in-out" },
          xaxis: {
            gridcolor: "rgba(20,29,33,0.12)",
            zerolinecolor: "rgba(20,29,33,0.3)",
            linecolor: "#141d21",
            tickfont: { color: "#3e4851" },
          },
          yaxis: {
            gridcolor: "rgba(20,29,33,0.12)",
            zerolinecolor: "rgba(20,29,33,0.3)",
            linecolor: "#141d21",
            tickfont: { color: "#3e4851" },
          },
          ...layout,
        }}
        config={{ responsive: true, displaylogo: false }}
        className="w-full"
        style={{ width: "100%", height }}
        useResizeHandler
      />
    </motion.div>
  );
}
