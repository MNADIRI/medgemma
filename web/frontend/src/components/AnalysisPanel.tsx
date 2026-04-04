import { useState } from "react";
import type { AnalysisResult, DiagnosisEntry, RoiData } from "../types";

interface Props {
  analysis: AnalysisResult;
}

/** Tier color + label mapping */
const TIER_STYLES: Record<string, { bg: string; border: string; label: string }> = {
  likely: { bg: "#1a2e1a", border: "#2d6a2d", label: "Likely" },
  possible: { bg: "#2a2a10", border: "#8a8a20", label: "Possible" },
  unlikely_but_to_exclude: { bg: "#2a1a1a", border: "#8a3030", label: "Unlikely but to exclude" },
};

function CollapsibleSection({
  title,
  defaultOpen = true,
  children,
  badge,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
  badge?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div style={{ borderRadius: 8, border: "1px solid #333", overflow: "hidden", marginBottom: 8 }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          width: "100%",
          padding: "10px 14px",
          background: "#1a1a1a",
          border: "none",
          color: "#ddd",
          fontSize: 13,
          fontWeight: 600,
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: 8,
          textAlign: "left",
        }}
      >
        <span style={{ fontSize: 10, color: "#888" }}>{open ? "▼" : "▶"}</span>
        {title}
        {badge && (
          <span
            style={{
              marginLeft: "auto",
              fontSize: 11,
              padding: "2px 8px",
              borderRadius: 4,
              background: "#4a90d9",
              color: "#fff",
            }}
          >
            {badge}
          </span>
        )}
      </button>
      {open && <div style={{ padding: "12px 14px", background: "#111" }}>{children}</div>}
    </div>
  );
}

export default function AnalysisPanel({ analysis }: Props) {
  const { localisation, aspect, diagnosis_entries, roi_data, raw_response } = analysis;
  const hasModelText = localisation || aspect || (diagnosis_entries && diagnosis_entries.length > 0);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        background: "#111",
        borderRadius: 8,
        border: "1px solid #333",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid #333",
          fontSize: 14,
          fontWeight: 600,
          color: "#ddd",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        Lesion Analysis
        <span style={{ fontSize: 11, color: "#4a90d9", fontWeight: 400 }}>MedGemma + MedSAM2</span>
      </div>

      {/* Scrollable content */}
      <div style={{ flex: 1, overflowY: "auto", padding: 12 }}>
        {/* Report section — model text */}
        <CollapsibleSection title="REPORT" defaultOpen={true}>
          {hasModelText ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 10, fontSize: 13, color: "#ccc" }}>
              {localisation && (
                <div>
                  <div style={{ fontSize: 11, color: "#4a90d9", fontWeight: 600, marginBottom: 4 }}>LOCALISATION</div>
                  <p style={{ margin: 0, lineHeight: 1.5 }}>{localisation}</p>
                </div>
              )}
              {aspect && (
                <div>
                  <div style={{ fontSize: 11, color: "#4a90d9", fontWeight: 600, marginBottom: 4 }}>ASPECT</div>
                  <p style={{ margin: 0, lineHeight: 1.5, whiteSpace: "pre-wrap" }}>{aspect}</p>
                </div>
              )}
            </div>
          ) : (
            <pre style={{ color: "#999", fontSize: 12, whiteSpace: "pre-wrap", margin: 0 }}>
              {raw_response || "No response from model."}
            </pre>
          )}
        </CollapsibleSection>

        {/* Diagnosis section */}
        <CollapsibleSection
          title="DIAGNOSIS"
          defaultOpen={true}
          badge={diagnosis_entries?.length ? `${diagnosis_entries.length} dx` : undefined}
        >
          {diagnosis_entries && diagnosis_entries.length > 0 ? (
            <DiagnosisCards entries={diagnosis_entries} />
          ) : analysis.diagnosis_text ? (
            <pre style={{ color: "#ccc", fontSize: 12, whiteSpace: "pre-wrap", margin: 0, lineHeight: 1.5 }}>
              {analysis.diagnosis_text}
            </pre>
          ) : (
            <p style={{ color: "#888", fontSize: 13, margin: 0 }}>No diagnosis parsed.</p>
          )}
        </CollapsibleSection>

        {/* ROI Data section — always reliable from pipeline */}
        {roi_data && (
          <CollapsibleSection title="ROI DATA" defaultOpen={false} badge="pipeline">
            <RoiDataPanel data={roi_data} />
          </CollapsibleSection>
        )}
      </div>
    </div>
  );
}

/* ── Diagnosis cards ─────────────────────────────────────────────────── */

function DiagnosisCards({ entries }: { entries: DiagnosisEntry[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {entries.map((dx, i) => {
        const tier = TIER_STYLES[dx.tier] || TIER_STYLES["possible"];
        return (
          <div
            key={i}
            style={{
              padding: "10px 12px",
              borderRadius: 6,
              background: tier.bg,
              border: `1px solid ${tier.border}`,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  padding: "2px 8px",
                  borderRadius: 4,
                  background: tier.border,
                  color: "#fff",
                  whiteSpace: "nowrap",
                }}
              >
                #{i + 1} {tier.label}
              </span>
              <span style={{ fontSize: 14, fontWeight: 600, color: "#eee" }}>{dx.label}</span>
            </div>

            {dx.supporting && (
              <div style={{ fontSize: 12, color: "#bbb", lineHeight: 1.5, marginBottom: 4 }}>
                <span style={{ color: "#6a6", fontWeight: 600 }}>Supporting: </span>
                {dx.supporting}
              </div>
            )}

            {dx.against && dx.against.toLowerCase() !== "none" && dx.against.toLowerCase() !== "none." && (
              <div style={{ fontSize: 12, color: "#999", lineHeight: 1.5 }}>
                <span style={{ color: "#a66", fontWeight: 600 }}>Against: </span>
                {dx.against}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ── ROI Data panel — quantitative pipeline data ─────────────────────── */

function RoiDataPanel({ data }: { data: RoiData }) {
  const { density, peri_lesional, morphometry, spatial } = data;
  const r = (v: number | null | undefined, d = 1) =>
    v != null ? v.toFixed(d) : "N/A";

  return (
    <div style={{ fontSize: 12, color: "#aaa", fontFamily: "monospace", lineHeight: 1.6 }}>
      {/* Density */}
      <div style={{ marginBottom: 8 }}>
        <div style={{ color: "#4a90d9", fontWeight: 600, marginBottom: 2, fontFamily: "sans-serif", fontSize: 11 }}>Density (eroded mask)</div>
        <div>mean={r(density.mean)} | median={r(density.median)} | sd={r(density.sd)} | min={r(density.min, 0)} | max={r(density.max, 0)}</div>
        <div>deciles: {density.deciles?.map((v, i) => `P${(i + 1) * 10}=${r(v)}`).join(" | ")}</div>
        <div>asymmetry={r(density.density_asymmetry)}</div>
      </div>

      {/* Peri-lesional */}
      <div style={{ marginBottom: 8 }}>
        <div style={{ color: "#4a90d9", fontWeight: 600, marginBottom: 2, fontFamily: "sans-serif", fontSize: 11 }}>Peri-lesional ring</div>
        <div>adjacent mean={r(peri_lesional.mean)} | sd={r(peri_lesional.sd)}</div>
        <div>
          delta_hu_median_parenchyma={" "}
          {peri_lesional.delta_hu_median_parenchyma != null
            ? `${peri_lesional.delta_hu_median_parenchyma >= 0 ? "+" : ""}${r(peri_lesional.delta_hu_median_parenchyma)}`
            : "N/A"}
        </div>
      </div>

      {/* Morphometry */}
      <div style={{ marginBottom: 8 }}>
        <div style={{ color: "#4a90d9", fontWeight: 600, marginBottom: 2, fontFamily: "sans-serif", fontSize: 11 }}>Morphometry (original mask)</div>
        <div>axes: {r(morphometry.major_axis_mm)} x {r(morphometry.minor_axis_mm)} mm | AR={r(morphometry.aspect_ratio, 2)}</div>
        <div>area={r(morphometry.area_mm2)} mm2 | perimeter={r(morphometry.perimeter_mm)} mm</div>
        <div>compactness={r(morphometry.compactness, 2)} | solidity={r(morphometry.solidity, 2)}</div>
        <div>eroded_area_fraction={r(morphometry.eroded_area_fraction, 2)}</div>
      </div>

      {/* Spatial */}
      <div>
        <div style={{ color: "#4a90d9", fontWeight: 600, marginBottom: 2, fontFamily: "sans-serif", fontSize: 11 }}>Spatial localization</div>
        <div>laterality: {spatial.laterality} | antero-posterior: {spatial.antero_posterior}</div>
      </div>
    </div>
  );
}
