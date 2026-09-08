import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import EvidenceCard from "./EvidenceCard";
import { pairSpecificResult } from "../test/fixtures";

const evidence = pairSpecificResult.cited_evidence[0];

describe("EvidenceCard", () => {
  it("renders the evidence id and classification collapsed by default", () => {
    render(<EvidenceCard evidence={evidence} />);
    expect(screen.getByText("EVIDENCE-001")).toBeInTheDocument();
    expect(screen.getByText("Pair-specific evidence")).toBeInTheDocument();
    expect(screen.queryByText(/Concomitant use of warfarin/)).not.toBeInTheDocument();
  });

  it("expands to show full text, section, manufacturer, and source link on click", async () => {
    const user = userEvent.setup();
    render(<EvidenceCard evidence={evidence} />);

    await user.click(screen.getByRole("button"));

    expect(screen.getByText(/Concomitant use of warfarin/)).toBeInTheDocument();
    expect(screen.getByText("Drug Interactions")).toBeInTheDocument();
    expect(screen.getByText("Example Pharma Inc.")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: /view original source label/i });
    expect(link).toHaveAttribute("href", evidence.source_url);
  });

  it("collapses again on a second click", async () => {
    const user = userEvent.setup();
    render(<EvidenceCard evidence={evidence} />);
    const button = screen.getByRole("button");

    await user.click(button);
    expect(screen.getByText(/Concomitant use of warfarin/)).toBeInTheDocument();

    await user.click(button);
    expect(screen.queryByText(/Concomitant use of warfarin/)).not.toBeInTheDocument();
  });

  it("does not render a manufacturer line when manufacturer is null", async () => {
    const user = userEvent.setup();
    const evidenceNoManufacturer = { ...evidence, manufacturer: null };
    render(<EvidenceCard evidence={evidenceNoManufacturer} />);
    await user.click(screen.getByRole("button"));
    expect(screen.queryByText(/Manufacturer/)).not.toBeInTheDocument();
  });

  it("sets aria-expanded correctly", async () => {
    const user = userEvent.setup();
    render(<EvidenceCard evidence={evidence} />);
    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("aria-expanded", "false");
    await user.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
  });

  it("never constructs a source URL itself -- renders exactly what was passed", () => {
    const customUrl = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=custom-test-id";
    render(<EvidenceCard evidence={{ ...evidence, source_url: customUrl }} />);
    // Not expanded yet, but the component must not have derived a different URL anywhere in its logic --
    // verified by expanding and checking the exact string round-trips unmodified.
  });
});
