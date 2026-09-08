import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Home from "./Home";
import * as api from "../services/api";
import {
  pairSpecificResult,
  insufficientEvidenceResult,
  evidenceOnlyFallbackResult,
  invalidInputResult,
} from "../test/fixtures";

describe("Home page", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(api, "searchDrugs").mockResolvedValue([]);
  });

  it("renders the header, subtitle, safety notice, and both checker sections", () => {
    render(<Home />);
    expect(screen.getByRole("heading", { name: "Drug Interaction Decision Support" })).toBeInTheDocument();
    expect(screen.getByText(/Evidence-backed drug interaction screening/)).toBeInTheDocument();
    expect(screen.getByRole("note", { name: /safety disclaimer/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Check Interaction" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Check Multiple Medications" })).toBeInTheDocument();
  });

  it("disables the Check Interaction button until both drug fields are filled", async () => {
    const user = userEvent.setup();
    render(<Home />);
    const button = screen.getByRole("button", { name: "Check Interaction" });
    expect(button).toBeDisabled();

    await user.type(screen.getByLabelText("Drug 1"), "warfarin");
    expect(button).toBeDisabled();

    await user.type(screen.getByLabelText("Drug 2"), "ibuprofen");
    expect(button).not.toBeDisabled();
  });

  it("does not submit on empty input (form validation)", async () => {
    vi.spyOn(api, "checkInteraction");
    const user = userEvent.setup();
    render(<Home />);
    // Button is disabled, so a click should not trigger a request at all.
    await user.click(screen.getByRole("button", { name: "Check Interaction" }));
    expect(api.checkInteraction).not.toHaveBeenCalled();
  });

  it("shows a loading state, then the successful result", async () => {
    let resolveFn;
    vi.spyOn(api, "checkInteraction").mockReturnValue(
      new Promise((resolve) => {
        resolveFn = resolve;
      })
    );
    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByLabelText("Drug 1"), "warfarin");
    await user.type(screen.getByLabelText("Drug 2"), "ibuprofen");
    await user.click(screen.getByRole("button", { name: "Check Interaction" }));

    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByText("Checking…")).toBeInTheDocument();

    await act(async () => {
      resolveFn(pairSpecificResult);
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(screen.getByText("Pair-specific evidence found")).toBeInTheDocument();
    });
  });

  it("displays insufficient evidence distinctly, not as 'no interaction'", async () => {
    vi.spyOn(api, "checkInteraction").mockResolvedValue(insufficientEvidenceResult);
    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByLabelText("Drug 1"), "warfarin");
    await user.type(screen.getByLabelText("Drug 2"), "metformin");
    await user.click(screen.getByRole("button", { name: "Check Interaction" }));

    await waitFor(() => {
      expect(screen.getByText("Insufficient evidence")).toBeInTheDocument();
    });
  });

  it("displays the evidence-only fallback when Gemini fails, without a blank screen", async () => {
    vi.spyOn(api, "checkInteraction").mockResolvedValue(evidenceOnlyFallbackResult);
    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByLabelText("Drug 1"), "warfarin");
    await user.type(screen.getByLabelText("Drug 2"), "ibuprofen");
    await user.click(screen.getByRole("button", { name: "Check Interaction" }));

    await waitFor(() => {
      expect(screen.getByText(/AI-generated explanation isn't available/)).toBeInTheDocument();
    });
    expect(screen.getByText("EVIDENCE-001")).toBeInTheDocument();
  });

  it("displays a same-medication message for invalid_input without an AI fallback note", async () => {
    vi.spyOn(api, "checkInteraction").mockResolvedValue(invalidInputResult);
    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByLabelText("Drug 1"), "warfarin");
    await user.type(screen.getByLabelText("Drug 2"), "Coumadin");
    await user.click(screen.getByRole("button", { name: "Check Interaction" }));

    await waitFor(() => {
      expect(screen.getByText("Same medication entered twice")).toBeInTheDocument();
    });
    expect(
      screen.getByText(
        "Warfarin and Coumadin resolve to the same medication. Please enter two different medications to check for an interaction."
      )
    ).toBeInTheDocument();
    expect(screen.queryByText(/AI-generated explanation isn't available/)).not.toBeInTheDocument();
  });

  it("shows a user-friendly error message on backend failure, with a retry option", async () => {
    const { ApiError } = api;
    vi.spyOn(api, "checkInteraction").mockRejectedValue(
      new ApiError("Unable to reach the server. Please check that the backend is running and try again.", {
        code: "network_error",
      })
    );
    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByLabelText("Drug 1"), "warfarin");
    await user.type(screen.getByLabelText("Drug 2"), "ibuprofen");
    await user.click(screen.getByRole("button", { name: "Check Interaction" }));

    await waitFor(() => {
      expect(screen.getByText(/unable to reach the server/i)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /try again/i })).toBeInTheDocument();
  });

  it("shows a rate-limit-specific friendly message", async () => {
    const { ApiError } = api;
    vi.spyOn(api, "checkInteraction").mockRejectedValue(
      new ApiError("Too many requests. Please wait a moment and try again.", { code: "rate_limit_exceeded", status: 429 })
    );
    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByLabelText("Drug 1"), "warfarin");
    await user.type(screen.getByLabelText("Drug 2"), "ibuprofen");
    await user.click(screen.getByRole("button", { name: "Check Interaction" }));

    await waitFor(() => {
      expect(screen.getByText(/too many requests/i)).toBeInTheDocument();
    });
  });

  it("retry button re-issues the request", async () => {
    const { ApiError } = api;
    vi.spyOn(api, "checkInteraction")
      .mockRejectedValueOnce(new ApiError("Unable to reach the server.", { code: "network_error" }))
      .mockResolvedValueOnce(pairSpecificResult);
    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByLabelText("Drug 1"), "warfarin");
    await user.type(screen.getByLabelText("Drug 2"), "ibuprofen");
    await user.click(screen.getByRole("button", { name: "Check Interaction" }));

    await waitFor(() => expect(screen.getByText(/unable to reach the server/i)).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: /try again/i }));

    await waitFor(() => {
      expect(screen.getByText("Pair-specific evidence found")).toBeInTheDocument();
    });
    expect(api.checkInteraction).toHaveBeenCalledTimes(2);
  });
});
