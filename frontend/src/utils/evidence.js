/**
 * utils/evidence.js
 * ====================
 * Presentation-only lookup tables mapping the backend's evidence_status
 * and classification VALUES (fixed strings defined in the backend —
 * see backend/docs/retrieval.md) to human-readable labels and a "tone"
 * used for styling.
 *
 * CRITICAL: this file does NOT decide what these statuses MEAN — it
 * only decides how to LABEL and STYLE values the backend already
 * computed. The frontend never invents, upgrades, or reinterprets an
 * evidence status or classification; it only presents the backend's
 * own words more readably. This is why "insufficient_evidence" gets a
 * neutral (not alarming, not reassuring) tone here — presenting it as
 * green/safe would visually imply "no interaction", which the backend
 * never asserts and the frontend must not imply either.
 */

export const ASSESSMENT_INFO = {
  pair_specific_evidence_found: {
    label: "Pair-specific evidence found",
    tone: "strong",
    description: "The retrieved evidence specifically connects these two medications.",
  },
  supporting_evidence_found: {
    label: "Supporting evidence found",
    tone: "moderate",
    description: "Relevant evidence was found, but it does not specifically connect these two medications.",
  },
  insufficient_evidence: {
    label: "Insufficient evidence",
    tone: "neutral",
    description: "Not enough evidence was found to assess this pair. This does not mean the medications are safe to combine.",
  },
  drug_not_found: {
    label: "Drug not found",
    tone: "warning",
    description: "One or both medication names could not be resolved.",
  },
  invalid_input: {
    label: "Same medication entered twice",
    tone: "warning",
    description: "Please enter two different medications to check for an interaction.",
  },
  retrieval_error: {
    label: "Retrieval error",
    tone: "error",
    description: "An error occurred while retrieving evidence. This is not a statement about the medications themselves.",
  },
};

export function getAssessmentInfo(status) {
  return (
    ASSESSMENT_INFO[status] || {
      label: status || "Unknown",
      tone: "neutral",
      description: "",
    }
  );
}

export const CLASSIFICATION_LABELS = {
  pair_specific_evidence: "Pair-specific evidence",
  class_level_evidence: "Class-level evidence",
  drug_specific_evidence: "Drug-specific evidence",
  general_label_evidence: "General label evidence",
};

export function getClassificationLabel(classification) {
  return CLASSIFICATION_LABELS[classification] || classification || "Evidence";
}
