import { useState } from "react";
import { getClassificationLabel } from "../utils/evidence";

const NOT_AVAILABLE = "Not available in retrieved source";

/**
 * EvidenceCard.jsx
 * ===================
 * One card per retrieved evidence item, letting the clinician/user
 * inspect exactly where the answer came from with full source transparency.
 *
 * REQUIREMENTS ENFORCED:
 * 1. Visibly exposes Evidence ID, Evidence classification, Source drug / label,
 *    FDA/DailyMed source metadata (Manufacturer, Section Code, RxCUI), Section name,
 *    actual retrieved evidence excerpt, and Source URL (if provided by backend).
 * 2. Clearly distinguishes pair-specific evidence from supporting evidence visually.
 * 3. Never invents text, metadata, or source URLs; displays "Not available in retrieved
 *    source" when a field is missing from the backend response.
 * 4. Renders backend source_url as-is without constructing or modifying it in React.
 */
export default function EvidenceCard({ evidence, defaultExpanded = true }) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const panelId = `evidence-panel-${evidence.evidence_id || "item"}`;
  const isPairSpecific = evidence.classification === "pair_specific_evidence";

  const classificationLabel = getClassificationLabel(evidence.classification) || NOT_AVAILABLE;
  const evidenceId = evidence.evidence_id || NOT_AVAILABLE;
  const drugName = evidence.drug_name || NOT_AVAILABLE;
  const sectionName = evidence.section_name || NOT_AVAILABLE;
  const sectionCode = evidence.section_code || NOT_AVAILABLE;
  const manufacturer = evidence.manufacturer || NOT_AVAILABLE;
  const rxcui = evidence.rxcui || NOT_AVAILABLE;
  const evidenceText = evidence.text || NOT_AVAILABLE;

  return (
    <div
      className={`evidence-card ${
        isPairSpecific ? "evidence-card--pair-specific" : "evidence-card--supporting"
      }`}
    >
      <button
        type="button"
        className="evidence-card-header"
        aria-expanded={expanded}
        aria-controls={panelId}
        onClick={() => setExpanded((v) => !v)}
      >
        <span className="evidence-id">{evidenceId}</span>
        <span
          className={`evidence-classification ${
            isPairSpecific ? "badge-pair-specific" : "badge-supporting"
          }`}
        >
          {classificationLabel}
        </span>
        {isPairSpecific && <span className="pair-match-badge">Direct Pair Match</span>}
        <span className="evidence-drug">{drugName}</span>
        <span className="chevron" aria-hidden="true">
          {expanded ? "\u25B2" : "\u25BC"}
        </span>
      </button>

      {expanded && (
        <div id={panelId} className="evidence-card-body">
          <div className="evidence-metadata-grid">
            <div className="metadata-item">
              <span className="metadata-label">Source Drug / Label:</span>
              <span className={`metadata-value ${!evidence.drug_name ? "unavailable" : ""}`}>
                {drugName}
              </span>
            </div>

            <div className="metadata-item">
              <span className="metadata-label">Section:</span>
              <span className={`metadata-value ${!evidence.section_name ? "unavailable" : ""}`}>
                {sectionName}
              </span>
            </div>

            <div className="metadata-item">
              <span className="metadata-label">Section Code:</span>
              <span className={`metadata-value ${!evidence.section_code ? "unavailable" : ""}`}>
                {sectionCode}
              </span>
            </div>

            <div className="metadata-item">
              <span className="metadata-label">Manufacturer / Labeler:</span>
              <span className={`metadata-value ${!evidence.manufacturer ? "unavailable" : ""}`}>
                {manufacturer}
              </span>
            </div>

            <div className="metadata-item">
              <span className="metadata-label">RxCUI:</span>
              <span className={`metadata-value ${!evidence.rxcui ? "unavailable" : ""}`}>
                {rxcui}
              </span>
            </div>

            <div className="metadata-item">
              <span className="metadata-label">Source URL:</span>
              <span className="metadata-value">
                {evidence.source_url ? (
                  <a
                    href={evidence.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="source-link"
                  >
                    View original source label
                  </a>
                ) : (
                  <span className="unavailable">{NOT_AVAILABLE}</span>
                )}
              </span>
            </div>
          </div>

          <div className="evidence-excerpt-section">
            <span className="metadata-label">Retrieved Evidence Text:</span>
            <p className={`evidence-text ${!evidence.text ? "unavailable" : ""}`}>
              {evidenceText}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
