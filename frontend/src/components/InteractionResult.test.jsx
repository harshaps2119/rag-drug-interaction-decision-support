import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import InteractionResult from "./InteractionResult";
import {
  pairSpecificResult,
  insufficientEvidenceResult,
  evidenceOnlyFallbackResult,
  drugNotFoundResult,
  invalidInputResult,
} from "../test/fixtures";

describe("InteractionResult", () => {
  it("renders nothing when result is null", () => {
    const { container } = render(<InteractionResult result={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the assessment badge, evidence summary, clinical effect, and evidence count for a pair-specific result", () => {
    render(<InteractionResult result={pairSpecificResult} />);
    expect(screen.getByText("Pair-specific evidence found")).toBeInTheDocument();
    expect(screen.getByText(/Warfarin combined with ibuprofen/)).toBeInTheDocument();
    expect(screen.getByText("Increased bleeding risk.")).toBeInTheDocument();
    expect(screen.getByText("Evidence (1)")).toBeInTheDocument();
    expect(screen.getByText("EVIDENCE-001")).toBeInTheDocument();
  });

  it('shows "Not stated in retrieved source" when severity is null -- never a blank field', () => {
    render(<InteractionResult result={pairSpecificResult} />);
    expect(screen.getByText("Not stated in retrieved source")).toBeInTheDocument();
  });

  it("shows the actual severity text when the backend provides one", () => {
    const withSeverity = { ...pairSpecificResult, severity: "Contraindicated" };
    render(<InteractionResult result={withSeverity} />);
    expect(screen.getByText("Contraindicated")).toBeInTheDocument();
    expect(screen.queryByText("Not stated in retrieved source")).not.toBeInTheDocument();
  });

  it("displays insufficient evidence with a neutral badge, never implying 'no interaction'", () => {
    render(<InteractionResult result={insufficientEvidenceResult} />);
    expect(screen.getByText("Insufficient evidence")).toBeInTheDocument();
    // Note: the component's OWN description text intentionally includes a negated
    // warning ("this does not mean the medications are safe to combine") -- so this
    // check looks for the phrase asserted AFFIRMATIVELY, not merely present as a
    // substring, which would false-positive on our own correctly-worded disclaimer.
    const resultText = document.body.textContent.toLowerCase();
    expect(resultText).not.toContain("no interaction exists");
    expect(resultText).not.toMatch(/(?<!not mean the medications are )safe to combine/);
  });

  it("shows 'No evidence was retrieved' when cited_evidence is empty", () => {
    render(<InteractionResult result={insufficientEvidenceResult} />);
    expect(screen.getByText("No evidence was retrieved for this pair.")).toBeInTheDocument();
  });

  it("evidence-only fallback: shows fallback note, evidence, but NO generated prose fields", () => {
    render(<InteractionResult result={evidenceOnlyFallbackResult} />);
    expect(screen.getByText(/AI-generated explanation isn't available/)).toBeInTheDocument();
    expect(screen.queryByText(/simulated outage/)).not.toBeInTheDocument();
    expect(screen.getByText("EVIDENCE-001")).toBeInTheDocument();
    // These prose fields must NOT render in evidence_only mode even though pairSpecificResult
    // has similar text -- the component gates on `mode`, not on field presence.
    expect(screen.queryByText("Evidence summary")).not.toBeInTheDocument();
    expect(screen.queryByText("Clinical effect")).not.toBeInTheDocument();
  });

  it("drug_not_found: shows the drug_not_found badge and fallback reason, no evidence section content", () => {
    render(<InteractionResult result={drugNotFoundResult} />);
    expect(screen.getByText("Drug not found")).toBeInTheDocument();
    expect(screen.queryByText(/drug_not_found/)).not.toBeInTheDocument();
    expect(screen.getByText("No evidence was retrieved for this pair.")).toBeInTheDocument();
  });

  it("invalid_input: explains that the same medication was entered twice without the AI fallback note", () => {
    render(<InteractionResult result={invalidInputResult} />);
    expect(screen.getByText("Same medication entered twice")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Warfarin and Coumadin resolve to the same medication. Please enter two different medications to check for an interaction."
      )
    ).toBeInTheDocument();
    expect(screen.queryByText(/AI-generated explanation isn't available/)).not.toBeInTheDocument();
  });

  it("renders limitations as a list when present", () => {
    render(<InteractionResult result={pairSpecificResult} />);
    expect(screen.getByText(pairSpecificResult.limitations[0])).toBeInTheDocument();
  });

  it("always renders the safety notice", () => {
    render(<InteractionResult result={pairSpecificResult} />);
    expect(screen.getByText(pairSpecificResult.safety_notice)).toBeInTheDocument();
  });

  it("applies a distinct tone class per assessment status (never the same tone for insufficient vs pair-specific)", () => {
    const { container: strongContainer } = render(<InteractionResult result={pairSpecificResult} />);
    const { container: neutralContainer } = render(<InteractionResult result={insufficientEvidenceResult} />);
    expect(strongContainer.querySelector(".tone-strong")).toBeInTheDocument();
    expect(neutralContainer.querySelector(".tone-neutral")).toBeInTheDocument();
  });
});
