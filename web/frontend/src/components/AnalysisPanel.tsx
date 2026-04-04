import { useCallback, useState } from "react";
import type { AnalysisResult } from "../types";

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
  const { report, diagnosis, chain_of_thought } = analysis;

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
        {/* Report section */}
        {report && !("_raw" in report) ? (
          <CollapsibleSection title="REPORT" defaultOpen={true}>
            <ReportContent report={report} />
          </CollapsibleSection>
        ) : (
          <CollapsibleSection title="REPORT" defaultOpen={true}>
            <p style={{ color: "#888", fontSize: 13, margin: 0 }}>
              Report parsing failed. See raw output below.
            </p>
          </CollapsibleSection>
        )}

        {/* Diagnosis section */}
        {diagnosis && !("_raw" in diagnosis) ? (
          <CollapsibleSection title="DIAGNOSIS" defaultOpen={true} badge={`${diagnosis.diagnostics?.length ?? 0} dx`}>
            <DiagnosisContent diagnosis={diagnosis} />
          </CollapsibleSection>
        ) : (
          <CollapsibleSection title="DIAGNOSIS" defaultOpen={true}>
            <p style={{ color: "#888", fontSize: 13, margin: 0 }}>
              Diagnosis parsing failed. See raw output below.
            </p>
          </CollapsibleSection>
        )}

        {/* Chain of thought (reasoning) — collapsed by default */}
        <CollapsibleSection title="REASONING" defaultOpen={false}>
          {chain_of_thought && !("_raw" in chain_of_thought) ? (
            <ReasoningContent cot={chain_of_thought} />
          ) : (
            <pre
              style={{
                color: "#999",
                fontSize: 12,
                fontFamily: "monospace",
                whiteSpace: "pre-wrap",
                margin: 0,
                maxHeight: 300,
                overflow: "auto",
              }}
            >
              {analysis.raw_response}
            </pre>
          )}
        </CollapsibleSection>
      </div>
    </div>
  );
}

/* ── Sub-components ──────────────────────────────────────────────────── */

function ReportContent({ report }: { report: NonNullable<AnalysisResult["report"]> }) {
  const { localisation, aspect, taille } = report;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, fontSize: 13, color: "#ccc" }}>
      {/* Localisation */}
      <div>
        <div style={{ fontSize: 11, color: "#4a90d9", fontWeight: 600, marginBottom: 4 }}>LOCALISATION</div>
        <p style={{ margin: 0, lineHeight: 1.5 }}>{localisation?.text || "N/A"}</p>
        {localisation?.laterality && (
          <div style={{ display: "flex", gap: 6, marginTop: 4, flexWrap: "wrap" }}>
            <Tag label={localisation.laterality} />
            {localisation.position && <Tag label={localisation.position} />}
            {localisation.organ && <Tag label={localisation.organ} color="#2a4a2a" />}
          </div>
        )}
      </div>

      {/* Aspect */}
      <div>
        <div style={{ fontSize: 11, color: "#4a90d9", fontWeight: 600, marginBottom: 4 }}>ASPECT</div>
        <p style={{ margin: 0, lineHeight: 1.5 }}>{aspect?.text || "N/A"}</p>
        {aspect && (
          <div style={{ display: "flex", gap: 6, marginTop: 4, flexWrap: "wrap" }}>
            <Tag label={`${aspect.density_class} (Δ${aspect.delta_hu >= 0 ? "+" : ""}${aspect.delta_hu?.toFixed?.(1) ?? "?"} HU)`} />
            <Tag label={aspect.homogeneity} />
            <Tag label={aspect.margins?.replace("_", " ")} />
            <Tag label={`${aspect.shape?.replace("_", " ")} (AR ${aspect.aspect_ratio?.toFixed?.(2) ?? "?"})`} />
          </div>
        )}
      </div>

      {/* Taille */}
      <div>
        <div style={{ fontSize: 11, color: "#4a90d9", fontWeight: 600, marginBottom: 4 }}>SIZE</div>
        <p style={{ margin: 0, lineHeight: 1.5 }}>{taille?.text || "N/A"}</p>
      </div>
    </div>
  );
}

function DiagnosisContent({ diagnosis }: { diagnosis: NonNullable<AnalysisResult["diagnosis"]> }) {
  const diagnostics = diagnosis.diagnostics || [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {diagnostics.map((dx, i) => {
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
                }}
              >
                #{dx.rank} {tier.label}
              </span>
              <span style={{ fontSize: 14, fontWeight: 600, color: "#eee" }}>{dx.label}</span>
            </div>

            {/* Supporting features */}
            {dx.supporting_features?.length > 0 && (
              <div style={{ marginBottom: 4 }}>
                <span style={{ fontSize: 11, color: "#6a6", fontWeight: 600 }}>Supporting: </span>
                <ul style={{ margin: "2px 0 0 16px", padding: 0, fontSize: 12, color: "#bbb", lineHeight: 1.5 }}>
                  {dx.supporting_features.map((f, j) => (
                    <li key={j}>{f}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Against features */}
            {dx.against_features?.length > 0 && dx.against_features[0] !== "" && (
              <div style={{ marginBottom: 4 }}>
                <span style={{ fontSize: 11, color: "#a66", fontWeight: 600 }}>Against: </span>
                <ul style={{ margin: "2px 0 0 16px", padding: 0, fontSize: 12, color: "#999", lineHeight: 1.5 }}>
                  {dx.against_features.map((f, j) => (
                    <li key={j}>{f}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Confidence rationale */}
            {dx.confidence_rationale && (
              <p style={{ margin: "4px 0 0", fontSize: 11, color: "#888", fontStyle: "italic" }}>
                {dx.confidence_rationale}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ReasoningContent({ cot }: { cot: Record<string, unknown> }) {
  // Display as formatted JSON for transparency
  return (
    <pre
      style={{
        color: "#999",
        fontSize: 11,
        fontFamily: "monospace",
        whiteSpace: "pre-wrap",
        margin: 0,
        maxHeight: 400,
        overflow: "auto",
        lineHeight: 1.4,
      }}
    >
      {JSON.stringify(cot, null, 2)}
    </pre>
  );
}

function Tag({ label, color = "#1a2a3a" }: { label: string; color?: string }) {
  return (
    <span
      style={{
        fontSize: 11,
        padding: "2px 8px",
        borderRadius: 4,
        background: color,
        color: "#aac",
        border: "1px solid #334",
      }}
    >
      {label}
    </span>
  );
}
