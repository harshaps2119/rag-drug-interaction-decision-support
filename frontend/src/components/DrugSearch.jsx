import { useId, useState } from "react";
import DrugInput from "./DrugInput";
import { useDrugSearch } from "../hooks/useDrugSearch";

/**
 * DrugSearch.jsx
 * =================
 * A labeled text input with live autocomplete suggestions from
 * GET /api/drugs/search (Phase 4's local knowledge base — no external
 * API call happens here or anywhere in this component).
 *
 * WHY NO RAW RXCUI IS SHOWN
 * ------------------------------
 * The backend's search result includes `rxcui` (used here only as the
 * React list key, an internal implementation detail) and `term_type`
 * (a short, meaningful label like "IN" or "BN"). The identifier itself
 * is never rendered to the user — only the normalized name and, where
 * present, the term type — per the project's explicit
 * "don't expose raw internal IDs unnecessarily" requirement.
 *
 * WHY AN UNMATCHED NAME CAN STILL BE SUBMITTED
 * --------------------------------------------------
 * This search box is a CONVENIENCE, not a gate: a name with no local
 * matches (e.g. a drug not yet ingested into the backend's knowledge
 * base) can still be typed and submitted — the backend's on-demand
 * ingestion (Phase 8) will attempt to resolve it live. The suggestion
 * list says so explicitly rather than implying the name is invalid.
 *
 * ACCESSIBILITY
 * ---------------
 * Implements the ARIA combobox pattern: `role="combobox"` on the
 * input, `role="listbox"`/`role="option"` on the suggestions, and full
 * keyboard support (Arrow Up/Down to move, Enter to select, Escape to
 * close) — not just mouse/click interaction.
 */
export default function DrugSearch({ label, value, onChange, placeholder }) {
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const { results, status } = useDrugSearch(value);
  const baseId = useId();
  const inputId = `${baseId}-input`;
  const listboxId = `${baseId}-listbox`;

  const showSuggestions = isOpen && value.trim().length >= 2;

  function selectResult(result) {
    onChange(result.normalized_name);
    setIsOpen(false);
    setActiveIndex(-1);
  }

  function handleKeyDown(event) {
    if (!showSuggestions || results.length === 0) return;

    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (event.key === "Enter" && activeIndex >= 0) {
      event.preventDefault();
      selectResult(results[activeIndex]);
    } else if (event.key === "Escape") {
      setIsOpen(false);
      setActiveIndex(-1);
    }
  }

  return (
    <div className="drug-search">
      <DrugInput
        id={inputId}
        label={label}
        value={value}
        onChange={(v) => {
          onChange(v);
          setIsOpen(true);
          setActiveIndex(-1);
        }}
        placeholder={placeholder}
        role="combobox"
        aria-expanded={showSuggestions}
        aria-controls={listboxId}
        aria-autocomplete="list"
        onFocus={() => setIsOpen(true)}
        onBlur={() => setTimeout(() => setIsOpen(false), 150)}
        onKeyDown={handleKeyDown}
      />

      {showSuggestions && (
        <ul id={listboxId} role="listbox" className="drug-search-suggestions" aria-label={`${label} suggestions`}>
          {status === "loading" && (
            <li className="suggestion-status" aria-live="polite">
              Searching…
            </li>
          )}
          {status === "success" && results.length === 0 && (
            <li className="suggestion-status">
              No matches in the local knowledge base — you can still submit this name.
            </li>
          )}
          {status === "success" &&
            results.map((result, index) => (
              <li
                key={result.rxcui}
                role="option"
                aria-selected={index === activeIndex}
                className={index === activeIndex ? "suggestion active" : "suggestion"}
                onMouseDown={() => selectResult(result)}
              >
                {result.normalized_name}
                {result.term_type && <span className="suggestion-meta"> · {result.term_type}</span>}
              </li>
            ))}
          {status === "error" && (
            <li className="suggestion-status">Search is temporarily unavailable — you can still type a name directly.</li>
          )}
        </ul>
      )}
    </div>
  );
}
