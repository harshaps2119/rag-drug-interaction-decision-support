/**
 * test/fixtures.js
 * ===================
 * Shared sample backend responses used across tests, shaped exactly
 * like the real backend schemas (backend/app/schemas/explanation.py,
 * evidence_assessment.py, api.py) — not simplified approximations.
 */

export const pairSpecificResult = {
  drug_a: { input_name: "warfarin", rxcui: "11289", normalized_name: "warfarin", resolved: true },
  drug_b: { input_name: "ibuprofen", rxcui: "5640", normalized_name: "ibuprofen", resolved: true },
  mode: "llm_grounded",
  interaction_assessment: "pair_specific_evidence_found",
  evidence_summary: "Warfarin combined with ibuprofen may increase the risk of bleeding.",
  clinical_effect: "Increased bleeding risk.",
  mechanism: null,
  severity: null,
  cited_evidence: [
    {
      evidence_id: "EVIDENCE-001",
      text: "Concomitant use of warfarin with NSAIDs such as ibuprofen may increase the risk of bleeding.",
      drug_name: "warfarin",
      rxcui: "11289",
      section_name: "Drug Interactions",
      section_code: "34073-7",
      manufacturer: "Example Pharma Inc.",
      source_url: "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=warfarin-setid",
      classification: "pair_specific_evidence",
    },
  ],
  limitations: ["Evidence is drawn only from each drug's own previously-ingested FDA label sections."],
  safety_notice: "Consult a pharmacist or physician before making treatment decisions.",
  fallback_reason: null,
  validation: { passed: true, fatal_issues: [], corrections: [], corrected_severity: null },
};

export const insufficientEvidenceResult = {
  drug_a: { input_name: "warfarin", rxcui: "11289", normalized_name: "warfarin", resolved: true },
  drug_b: { input_name: "metformin", rxcui: "6809", normalized_name: "metformin", resolved: true },
  mode: "llm_grounded",
  interaction_assessment: "insufficient_evidence",
  evidence_summary: "No relevant evidence was retrieved for this drug pair.",
  clinical_effect: null,
  mechanism: null,
  severity: null,
  cited_evidence: [],
  limitations: ["This retrieval layer does not itself determine whether an interaction exists."],
  safety_notice: "Consult a pharmacist or physician before making treatment decisions.",
  fallback_reason: null,
  validation: { passed: true, fatal_issues: [], corrections: [], corrected_severity: null },
};

export const evidenceOnlyFallbackResult = {
  drug_a: { input_name: "warfarin", rxcui: "11289", normalized_name: "warfarin", resolved: true },
  drug_b: { input_name: "ibuprofen", rxcui: "5640", normalized_name: "ibuprofen", resolved: true },
  mode: "evidence_only",
  interaction_assessment: "pair_specific_evidence_found",
  evidence_summary: null,
  clinical_effect: null,
  mechanism: null,
  severity: null,
  cited_evidence: [
    {
      evidence_id: "EVIDENCE-001",
      text: "Concomitant use of warfarin with NSAIDs such as ibuprofen may increase the risk of bleeding.",
      drug_name: "warfarin",
      rxcui: "11289",
      section_name: "Drug Interactions",
      section_code: "34073-7",
      manufacturer: null,
      source_url: "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=warfarin-setid",
      classification: "pair_specific_evidence",
    },
  ],
  limitations: ["Standard limitation text."],
  safety_notice: "Consult a pharmacist or physician before making treatment decisions.",
  fallback_reason: "LLM unavailable (Gemini): simulated outage",
  validation: null,
};

export const drugNotFoundResult = {
  drug_a: { input_name: "notarealdrug1", rxcui: null, normalized_name: null, resolved: false },
  drug_b: { input_name: "notarealdrug2", rxcui: null, normalized_name: null, resolved: false },
  mode: "evidence_only",
  interaction_assessment: "drug_not_found",
  evidence_summary: null,
  clinical_effect: null,
  mechanism: null,
  severity: null,
  cited_evidence: [],
  limitations: [],
  safety_notice: "Consult a pharmacist or physician before making treatment decisions.",
  fallback_reason: "Cannot generate an explanation: retrieval result was 'drug_not_found'.",
  validation: null,
};

export const invalidInputResult = {
  drug_a: { input_name: "warfarin", rxcui: "11289", normalized_name: "warfarin", resolved: true },
  drug_b: { input_name: "Coumadin", rxcui: "11289", normalized_name: "warfarin", resolved: true },
  mode: "evidence_only",
  interaction_assessment: "invalid_input",
  evidence_summary: null,
  clinical_effect: null,
  mechanism: null,
  severity: null,
  cited_evidence: [],
  limitations: [],
  safety_notice: "Consult a pharmacist or physician before making treatment decisions.",
  fallback_reason: "Cannot generate an explanation: retrieval result was 'invalid_input'.",
  validation: null,
};

export const drugSearchResults = [
  { rxcui: "11289", normalized_name: "warfarin", term_type: "IN" },
  { rxcui: "153010", normalized_name: "warfarin sodium", term_type: "PIN" },
];

export function multiDrugResponse(pairs) {
  return {
    pairs: pairs.map(([a, b, result]) => ({ drug_a: a, drug_b: b, result })),
    total_pairs: pairs.length,
    request_id: "test-request-id",
  };
}
