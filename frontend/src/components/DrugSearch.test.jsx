import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DrugSearch from "./DrugSearch";
import * as api from "../services/api";
import { drugSearchResults } from "../test/fixtures";

describe("DrugSearch", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders a labeled text input", () => {
    render(<DrugSearch label="Drug 1" value="" onChange={() => {}} />);
    expect(screen.getByLabelText("Drug 1")).toBeInTheDocument();
  });

  it("calls onChange as the user types", async () => {
    const user = userEvent.setup();
    const handleChange = vi.fn();
    function Wrapper() {
      return <DrugSearch label="Drug 1" value="" onChange={handleChange} />;
    }
    render(<Wrapper />);
    await user.type(screen.getByLabelText("Drug 1"), "w");
    expect(handleChange).toHaveBeenCalledWith("w");
  });

  it("does not search for a query under 2 characters", async () => {
    const searchSpy = vi.spyOn(api, "searchDrugs");
    render(<DrugSearch label="Drug 1" value="w" onChange={() => {}} />);
    await act(async () => {
      await new Promise((r) => setTimeout(r, 350));
    });
    expect(searchSpy).not.toHaveBeenCalled();
  });

  it("shows matching suggestions after typing, without exposing raw rxcui text", async () => {
    vi.spyOn(api, "searchDrugs").mockResolvedValue(drugSearchResults);

    function Controlled() {
      const [value, setValue] = require("react").useState("warf");
      return <DrugSearch label="Drug 1" value={value} onChange={setValue} />;
    }
    render(<Controlled />);
    await userEvent.click(screen.getByLabelText("Drug 1"));

    await waitFor(() => {
      expect(screen.getByText("warfarin")).toBeInTheDocument();
    });
    expect(screen.getByText("warfarin sodium")).toBeInTheDocument();
    // rxcui values themselves must never appear as visible text.
    expect(screen.queryByText("11289")).not.toBeInTheDocument();
    expect(screen.queryByText("153010")).not.toBeInTheDocument();
  });

  it("shows a helpful message (not an error) when there are no local matches", async () => {
    vi.spyOn(api, "searchDrugs").mockResolvedValue([]);

    function Controlled() {
      const [value, setValue] = require("react").useState("xyzunlisted");
      return <DrugSearch label="Drug 1" value={value} onChange={setValue} />;
    }
    render(<Controlled />);
    await userEvent.click(screen.getByLabelText("Drug 1"));

    await waitFor(() => {
      expect(screen.getByText(/no matches in the local knowledge base/i)).toBeInTheDocument();
    });
  });

  it("selecting a suggestion calls onChange with the normalized name", async () => {
    vi.spyOn(api, "searchDrugs").mockResolvedValue(drugSearchResults);
    const user = userEvent.setup();
    const handleChange = vi.fn();

    function Controlled() {
      const [value, setValue] = require("react").useState("warf");
      return (
        <DrugSearch
          label="Drug 1"
          value={value}
          onChange={(v) => {
            setValue(v);
            handleChange(v);
          }}
        />
      );
    }
    render(<Controlled />);
    await user.click(screen.getByLabelText("Drug 1"));

    await waitFor(() => expect(screen.getByText("warfarin")).toBeInTheDocument());
    await user.click(screen.getByText("warfarin"));

    expect(handleChange).toHaveBeenCalledWith("warfarin");
  });

  it("has combobox and listbox ARIA roles for accessibility", async () => {
    vi.spyOn(api, "searchDrugs").mockResolvedValue(drugSearchResults);
    function Controlled() {
      const [value, setValue] = require("react").useState("warf");
      return <DrugSearch label="Drug 1" value={value} onChange={setValue} />;
    }
    render(<Controlled />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
    await userEvent.click(screen.getByLabelText("Drug 1"));
    await waitFor(() => expect(screen.getByRole("listbox")).toBeInTheDocument());
  });
});
