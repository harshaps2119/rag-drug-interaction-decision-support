import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import MultiDrugChecker from "./MultiDrugChecker";
import * as api from "../services/api";
import { pairSpecificResult, multiDrugResponse } from "../test/fixtures";

describe("MultiDrugChecker", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the add-medication input and an initially-disabled submit button", () => {
    render(<MultiDrugChecker />);
    expect(screen.getByLabelText("Add a medication")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /check 0 pairs/i })).toBeDisabled();
  });

  it("adds a medication to the list via the Add button", async () => {
    const user = userEvent.setup();
    render(<MultiDrugChecker />);
    await user.type(screen.getByLabelText("Add a medication"), "warfarin");
    await user.click(screen.getByRole("button", { name: "Add" }));
    expect(screen.getByText("warfarin")).toBeInTheDocument();
  });

  it("adds a medication via pressing Enter", async () => {
    const user = userEvent.setup();
    render(<MultiDrugChecker />);
    await user.type(screen.getByLabelText("Add a medication"), "ibuprofen{Enter}");
    expect(screen.getByText("ibuprofen")).toBeInTheDocument();
  });

  it("prevents duplicate medications (case-insensitive) and shows a warning", async () => {
    const user = userEvent.setup();
    render(<MultiDrugChecker />);
    const input = screen.getByLabelText("Add a medication");

    await user.type(input, "warfarin{Enter}");
    await user.type(input, "Warfarin{Enter}");

    expect(screen.getAllByText("warfarin")).toHaveLength(1);
    expect(screen.getByText(/already in the list/i)).toBeInTheDocument();
  });

  it("removes a medication from the list", async () => {
    const user = userEvent.setup();
    render(<MultiDrugChecker />);
    await user.type(screen.getByLabelText("Add a medication"), "warfarin{Enter}");
    expect(screen.getByText("warfarin")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /remove warfarin/i }));
    expect(screen.queryByText("warfarin")).not.toBeInTheDocument();
  });

  it("enables submit only once at least 2 medications are added, and shows the correct pair count", async () => {
    const user = userEvent.setup();
    render(<MultiDrugChecker />);
    const input = screen.getByLabelText("Add a medication");

    await user.type(input, "warfarin{Enter}");
    expect(screen.getByRole("button", { name: /check 0 pairs/i })).toBeDisabled();

    await user.type(input, "ibuprofen{Enter}");
    expect(screen.getByRole("button", { name: /check 1 pair$/i })).not.toBeDisabled();

    await user.type(input, "aspirin{Enter}");
    expect(screen.getByRole("button", { name: /check 3 pairs/i })).not.toBeDisabled();
  });

  it("submits the drug list and displays grouped pair results", async () => {
    vi.spyOn(api, "checkMultipleInteractions").mockResolvedValue(
      multiDrugResponse([["warfarin", "ibuprofen", pairSpecificResult]])
    );
    const user = userEvent.setup();
    render(<MultiDrugChecker />);
    const input = screen.getByLabelText("Add a medication");
    await user.type(input, "warfarin{Enter}");
    await user.type(input, "ibuprofen{Enter}");

    await user.click(screen.getByRole("button", { name: /check 1 pair$/i }));

    await waitFor(() => {
      expect(screen.getByText("warfarin + ibuprofen")).toBeInTheDocument();
    });
    expect(screen.getByText("Pair-specific evidence found")).toBeInTheDocument();
    expect(api.checkMultipleInteractions).toHaveBeenCalledWith(["warfarin", "ibuprofen"]);
  });

  it("shows a loading state while the multi-check request is in flight", async () => {
    let resolveFn;
    vi.spyOn(api, "checkMultipleInteractions").mockReturnValue(
      new Promise((resolve) => {
        resolveFn = resolve;
      })
    );
    const user = userEvent.setup();
    render(<MultiDrugChecker />);
    const input = screen.getByLabelText("Add a medication");
    await user.type(input, "warfarin{Enter}");
    await user.type(input, "ibuprofen{Enter}");
    await user.click(screen.getByRole("button", { name: /check 1 pair$/i }));

    expect(screen.getByRole("status")).toBeInTheDocument();
    await act(async () => {
      resolveFn(multiDrugResponse([]));
      await Promise.resolve();
    });
  });

  it("shows an error message and a retry button if the request fails", async () => {
    const { ApiError } = api;
    vi.spyOn(api, "checkMultipleInteractions").mockRejectedValue(
      new ApiError("Too many requests. Please wait a moment and try again.", { code: "rate_limit_exceeded", status: 429 })
    );
    const user = userEvent.setup();
    render(<MultiDrugChecker />);
    const input = screen.getByLabelText("Add a medication");
    await user.type(input, "warfarin{Enter}");
    await user.type(input, "ibuprofen{Enter}");
    await user.click(screen.getByRole("button", { name: /check 1 pair$/i }));

    await waitFor(() => {
      expect(screen.getByText(/too many requests/i)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /try again/i })).toBeInTheDocument();
  });
});
