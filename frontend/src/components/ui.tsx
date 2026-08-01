"use client";

import { motion } from "framer-motion";
import { BatteryCharging, Droplets, Leaf, PlugZap, SunMedium, Wind } from "lucide-react";

import { cn } from "@/lib/utils";

export function PixelPanel({
  className,
  children,
  accent = "default",
}: {
  className?: string;
  children: React.ReactNode;
  accent?: "default" | "blue" | "green";
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: "easeOut" }}
      className={cn(
        "pixel-panel relative overflow-hidden p-5",
        accent === "blue" && "pixel-panel-blue",
        accent === "green" && "pixel-panel-green",
        className,
      )}
    >
      {children}
    </motion.div>
  );
}

export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return <PixelPanel className={className}>{children}</PixelPanel>;
}

export function PixelButton({
  className,
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { children: React.ReactNode }) {
  return (
    <button
      className={cn(
        "pixel-button bg-[#00aaff] px-4 py-2 text-sm font-bold uppercase tracking-wide text-white hover:bg-[#016e00] disabled:cursor-not-allowed disabled:opacity-60",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

export function PageShell({ children }: { children: React.ReactNode }) {
  return <div className="relative min-h-screen overflow-hidden">{children}</div>;
}

export function Section({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <motion.section
      className="space-y-5"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.32, ease: "easeOut" }}
    >
      <div className="pixel-panel bg-white/80 p-5">
        <p className="mb-1 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.18em] text-[#016e00]">
          <span className="h-3 w-3 border border-[#141d21] bg-[#75fd43]" />
          System module
        </p>
        <h2 className="font-display text-3xl font-bold uppercase text-[#016e00] md:text-4xl">{title}</h2>
        {description ? <p className="mt-2 max-w-3xl text-sm leading-6 text-[#3e4851]">{description}</p> : null}
      </div>
      {children}
    </motion.section>
  );
}

export function Kpi({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <PixelPanel className="border-b-[7px] border-b-[#00aaff] bg-white/92 text-center">
      <div className="text-xs font-bold uppercase tracking-wide text-[#3e4851]/80">{label}</div>
      <div className="font-display mt-2 break-words text-2xl font-bold text-[#016e00]">{value}</div>
      {hint ? <div className="mt-2 text-xs leading-5 text-[#3e4851]">{hint}</div> : null}
    </PixelPanel>
  );
}

export function SimpleTable({
  columns,
  rows,
}: {
  columns: string[];
  rows: Record<string, string | number | null | undefined>[];
}) {
  return (
    <div className="pixel-panel overflow-x-auto bg-white/90 p-0">
      <table className="w-full min-w-[560px] text-left text-sm">
        <thead className="bg-[#00aaff] text-white">
          <tr>
            {columns.map((column) => (
              <th className="border-b-[3px] border-[#141d21] px-4 py-3 font-bold uppercase tracking-wide" key={column}>
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className={index % 2 ? "bg-[#f4faff]" : "bg-white"}>
              {columns.map((column) => (
                <td className="border-t border-[#d2dbe1] px-4 py-3 text-[#141d21]" key={column}>
                  {row[column] ?? "n/a"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children?: React.ReactNode;
}) {
  return (
    <PixelPanel className="flex flex-col gap-4 bg-white/88 md:flex-row md:items-end md:justify-between" accent="green">
      <div>
        <p className="mb-1 text-xs font-bold uppercase tracking-[0.22em] text-[#0061a5]">Eco-t Energy Intelligence</p>
        <h1 className="font-display text-3xl font-bold uppercase leading-tight text-[#016e00] md:text-5xl">{title}</h1>
        <p className="mt-2 max-w-2xl text-sm text-[#3e4851]">{subtitle}</p>
      </div>
      {children}
    </PixelPanel>
  );
}

export function DatasetSelector({
  value,
  datasets,
  onChange,
}: {
  value: string;
  datasets: { key: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="block min-w-44 text-xs font-bold uppercase tracking-wide text-[#141d21]">
      Dataset
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-2 w-full border-[3px] border-[#141d21] bg-white px-3 py-2 text-sm font-bold text-[#016e00] shadow-[4px_4px_0_rgba(20,29,33,0.2)] outline-none focus:ring-4 focus:ring-[#00aaff]/30"
      >
        {datasets.map((dataset) => (
          <option key={dataset.key} value={dataset.key}>
            {dataset.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function EnergyFigurine({
  type = "leaf",
  className,
  label,
}: {
  type?: "wind" | "solar" | "hydro" | "leaf" | "grid" | "operator";
  className?: string;
  label?: string;
}) {
  const iconMap = {
    wind: Wind,
    solar: SunMedium,
    hydro: Droplets,
    leaf: Leaf,
    grid: PlugZap,
    operator: BatteryCharging,
  };
  const Icon = iconMap[type];
  const color = type === "solar" ? "#f1c100" : type === "hydro" || type === "wind" ? "#00aaff" : "#016e00";

  return (
    <div className={cn("pointer-events-none inline-flex flex-col items-center gap-1 animate-pixel-float", className)}>
      <div className="pixel-panel flex h-16 w-16 items-center justify-center bg-white/85 p-0">
        <Icon color={color} size={34} strokeWidth={2.8} />
      </div>
      {label ? <span className="bg-white/80 px-1 text-[10px] font-bold uppercase tracking-wide text-[#3e4851]">{label}</span> : null}
    </div>
  );
}

export function Preloader({ visible }: { visible: boolean }) {
  const icons = [
    { type: "wind" as const, label: "WIND_GEN" },
    { type: "solar" as const, label: "SOLAR_CELL" },
    { type: "hydro" as const, label: "HYDRO_FLUX" },
    { type: "leaf" as const, label: "BIO_PULSE" },
  ];

  if (!visible) return null;

  return (
    <motion.div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-[#f4faff]/88 p-4 backdrop-blur-md"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.18 }}
    >
      <div className="absolute inset-0 dither-bg opacity-40" />
      <motion.div
        className="pixel-panel relative w-full max-w-[520px] bg-white p-6 pt-12"
        initial={{ y: 18, scale: 0.98 }}
        animate={{ y: 0, scale: 1 }}
        transition={{ duration: 0.28, ease: "easeOut" }}
      >
        <div className="absolute -left-[3px] -right-[3px] -top-[3px] flex h-9 items-center justify-between border-[3px] border-[#141d21] bg-[#016e00] px-4 text-white">
          <span className="text-xs font-bold uppercase tracking-[0.18em]">Eco-t System Boot</span>
          <div className="flex gap-1">
            <span className="h-3 w-3 bg-white/30" />
            <span className="h-3 w-3 bg-white/60" />
            <span className="h-3 w-3 bg-white" />
          </div>
        </div>
        <div className="flex min-h-[260px] flex-col items-center justify-center">
          <div className="relative mb-8 h-24 w-24">
            {icons.map((item, index) => (
              <motion.div
                key={item.label}
                className="absolute inset-0 flex items-center justify-center"
                animate={{ opacity: [0, 1, 1, 0], scale: [0.92, 1, 1, 0.92] }}
                transition={{ duration: 4, repeat: Infinity, delay: index, times: [0, 0.1, 0.22, 0.32] }}
              >
                <EnergyFigurine type={item.type} label={item.label} />
              </motion.div>
            ))}
          </div>
          <p className="font-display text-center text-2xl font-bold italic text-[#141d21]">Sustainable Energy Intelligence</p>
          <div className="mt-8 w-full space-y-3">
            <div className="flex justify-between text-xs font-bold uppercase tracking-wide text-[#016e00]">
              <span>System_loading_state</span>
              <span>100%</span>
            </div>
            <div className="pixel-panel flex h-8 gap-1 bg-white p-1">
              {Array.from({ length: 16 }).map((_, index) => (
                <span
                  key={index}
                  className="h-full flex-1 bg-[#75fd43] animate-segmented-pulse"
                  style={{ animationDelay: `${index * 55}ms` }}
                />
              ))}
            </div>
            <p className="text-center text-[10px] font-bold uppercase tracking-[0.16em] text-[#3e4851]">Establishing connection to grid...</p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
