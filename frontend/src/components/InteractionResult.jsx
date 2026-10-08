import EvidenceCard from "./EvidenceCard";
import { getAssessmentInfo } from "../utils/evidence";

/**
 * InteractionResult.jsx
 * ========================
 * Renders one ExplanationResponse from the backend — the single most
 * safety-sensitive component in this frontend, because it's the only
 * place a person actually reads the result.
 *
 * RULES THIS COMPONENT FOLLOWS, MATCHING THE BACKEND'S OWN RULES
 * --------------------------------------------------------------------
 * 1. The assessment badge label/tone comes from utils/evidence.js,
 *    which maps the backend's fixed status strings to labels — it
 *    never invents a new status or reinterprets one.
 * 2. "insufficient_evidence" and "drug_not_found" are NEVER styled or
 *    worded as "no interaction" / "safe" — see the neutral/warning
 *    tones in utils/evidence.js and the explicit description text.
 * 3. Prose fields (evidence_summary, clinical_effect, mechanism) are
 *    only shown in `mode === "llm_grounded"`. In `mode === "evidence_only"`
 *    (the LLM failed, or its response failed grounding validation), NO
 *    generated prose is shown — only the fallback_reason and the raw
 *    retrieved evidence, exactly matching what the backend actually
 *    decided to show, never papering over the gap with placeholder text.
 * 4. Severity ALWAYS renders something — "Not stated in retrieved
 *    source" when the backend's severity field is null — so the UI
 *    never shows a blank space that looks like missing/broken UI data
 *    (an explicit design requirement).
 * 5. Evidence presentation:
 *    - Pair-specific evidence is prioritized and presented first.
 *    - Lower-specificity evidence is grouped under "Additional supporting evidence".
 *    - Every evidence item visibly exposes its excerpt and FDA metadata without
 *      requiring user clicks.
 */
export default function InteractionResult({ result }) {
  if (!result) return null;

  const assessment = getAssessmentInfo(result.interaction_assessment);
  const isFallback = result.mode === "evidence_only";
  const isInvalidInput = result.interaction_assessment === "invalid_input";
  const evidenceItems = result.cited_evidence || [];

  const pairEvidence = evidenceItems.filter(
    (item) => item.classification === "pair_specific_evidence"
  );
  const supportingEvidence = evidenceItems.filter(
    (item) => item.classification !== "pair_specific_evidence"
  );

  const drugAName = result?.drug_a?.input_name || result?.drug_a?.normalized_name || "First medication";
  const drugBName = result?.drug_b?.input_name || result?.drug_b?.normalized_name || "second medication";
  const formattedA = drugAName.charAt(0).toUpperCase() + drugAName.slice(1);
  const formattedB = drugBName.charAt(0).toUpperCase() + drugBName.slice(1);
  const assessmentDescription = isInvalidInput
    ? `${formattedA} and ${formattedB} resolve to the same medication. ${assessment.description}`
    : assessment.description;

  return (
    <section className="interaction-result" aria-live="polite">
      <div className={`assessment-badge tone-${assessment.tone}`}>
        <span className="assessment-label">{assessment.label}</span>
        {assessmentDescription && <p className="assessment-description">{assessmentDescription}</p>}
      </div>

      {isFallback && !isInvalidInput && (
        <p className="fallback-note">
          An AI-generated explanation isn't available for this result.
          The evidence retrieved from FDA labels is shown below.
        </p>
      )}

      {!isFallback && result.evidence_summary && (
        <div className="result-section">
          <h3>Evidence summary</h3>
          <p>{result.evidence_summary}</p>
        </div>
      )}

      {!isFallback && result.clinical_effect && (
        <div className="result-section">
          <h3>Clinical effect</h3>
          <p>{result.clinical_effect}</p>
        </div>
      )}

      {!isFallback && result.mechanism && (
        <div className="result-section">
          <h3>Mechanism</h3>
          <p>{result.mechanism}</p>
        </div>
      )}

      <div className="result-section">
        <h3>Severity</h3>
        <p>{result.severity || "Not stated in retrieved source"}</p>
      </div>

      {result.limitations && result.limitations.length > 0 && (
        <div className="result-section">
          <h3>Limitations</h3>
          <ul>
            {result.limitations.map((limitation, index) => (
              <li key={index}>{limitation}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="result-section evidence-section">
        <h3>Evidence ({evidenceItems.length})</h3>
        {evidenceItems.length === 0 ? (
          <p>No evidence was retrieved for this pair.</p>
        ) : (
          <div className="evidence-groups">
            {pairEvidence.length > 0 && (
              <div className="evidence-group evidence-group--pair-specific">
                <div className="evidence-group-header">
                  <h4 className="evidence-group-title">
                    Pair-specific evidence ({pairEvidence.length})
                  </h4>
                  <p className="evidence-group-subtitle">
                    Evidence citing both medications together in FDA drug labeling.
                  </p>
                </div>
                <div className="evidence-list">
                  {pairEvidence.map((evidence) => (
                    <EvidenceCard key={evidence.evidence_id} evidence={evidence} />
                  ))}
                </div>
              </div>
            )}

            {supportingEvidence.length > 0 && (
              <div className="evidence-group evidence-group--supporting">
                <div className="evidence-group-header">
                  <h4 className="evidence-group-title">
                    Additional supporting evidence ({supportingEvidence.length})
                  </h4>
                  <p className="evidence-group-subtitle">
                    Relevant single-drug, class-level, or general FDA labeling context without direct pair mention.
                  </p>
                </div>
                <div className="evidence-list">
                  {supportingEvidence.map((evidence) => (
                    <EvidenceCard key={evidence.evidence_id} evidence={evidence} />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {result.safety_notice && <p className="result-safety-notice">{result.safety_notice}</p>}
    </section>
  );
}
