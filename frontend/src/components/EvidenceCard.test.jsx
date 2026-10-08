import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import EvidenceCard from "./EvidenceCard";
import { pairSpecificResult } from "../test/fixtures";

const evidence = pairSpecificResult.cited_evidence[0];

describe("EvidenceCard", () => {
  it("renders the evidence id, classification, source drug, metadata, and excerpt visibly by default", () => {
    render(<EvidenceCard evidence={evidence} />);
    expect(screen.getByText("EVIDENCE-001")).toBeInTheDocument();
    expect(screen.getByText("Pair-specific evidence")).toBeInTheDocument();
    expect(screen.getByText(/Concomitant use of warfarin/)).toBeInTheDocument();
    expect(screen.getByText("Drug Interactions")).toBeInTheDocument();
    expect(screen.getByText("Example Pharma Inc.")).toBeInTheDocument();
    expect(screen.getByText("34073-7")).toBeInTheDocument();
    expect(screen.getByText("11289")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: /view original source label/i });
    expect(link).toHaveAttribute("href", evidence.source_url);
  });

  it("collapses to hide excerpt and metadata on click, and expands on a second click", async () => {
    const user = userEvent.setup();
    render(<EvidenceCard evidence={evidence} />);
    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/Concomitant use of warfarin/)).toBeInTheDocument();

    await user.click(button);
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText(/Concomitant use of warfarin/)).not.toBeInTheDocument();

    await user.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/Concomitant use of warfarin/)).toBeInTheDocument();
  });

  it("displays 'Not available in retrieved source' when manufacturer or other metadata is null", () => {
    const evidenceNoManufacturer = { ...evidence, manufacturer: null };
    render(<EvidenceCard evidence={evidenceNoManufacturer} />);
    expect(screen.getByText("Not available in retrieved source")).toBeInTheDocument();
  });

  it("displays 'Not available in retrieved source' when source_url is null without inventing a URL", () => {
    const evidenceNoUrl = { ...evidence, source_url: null };
    render(<EvidenceCard evidence={evidenceNoUrl} />);
    expect(screen.queryByRole("link", { name: /view original source label/i })).not.toBeInTheDocument();
    expect(screen.getByText("Not available in retrieved source")).toBeInTheDocument();
  });

  it("visually distinguishes pair-specific evidence from supporting evidence", () => {
    const { container: pairContainer } = render(<EvidenceCard evidence={evidence} />);
    expect(pairContainer.querySelector(".evidence-card--pair-specific")).toBeInTheDocument();
    expect(pairContainer.querySelector(".pair-match-badge")).toBeInTheDocument();

    const supportingEvidence = { ...evidence, classification: "drug_specific_evidence" };
    const { container: supportingContainer } = render(<EvidenceCard evidence={supportingEvidence} />);
    expect(supportingContainer.querySelector(".evidence-card--supporting")).toBeInTheDocument();
    expect(supportingContainer.querySelector(".pair-match-badge")).not.toBeInTheDocument();
  });

  it("never constructs a source URL itself -- renders exactly what was passed", () => {
    const customUrl = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=custom-test-id";
    render(<EvidenceCard evidence={{ ...evidence, source_url: customUrl }} />);
    const link = screen.getByRole("link", { name: /view original source label/i });
    expect(link).toHaveAttribute("href", customUrl);
  });
});
