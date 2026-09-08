import { useState } from "react";
import DrugInput from "./DrugInput";
import InteractionResult from "./InteractionResult";
import LoadingState from "./LoadingState";
import ErrorMessage from "./ErrorMessage";
import { useMultiInteractionCheck } from "../hooks/useMultiInteractionCheck";

/**
 * MultiDrugChecker.jsx
 * =======================
 * "Check Multiple Medications" — lets a user build a list of drug
 * names, then submits the whole list to POST /api/interaction/check-multiple,
 * which generates every unique pair on the BACKEND (Phase 6) and
 * returns one result per pair. This component never generates pairs
 * itself — it only sends the flat list and renders whatever pairs come
 * back.
 *
 * DUPLICATE PREVENTION
 * -----------------------
 * Checked case-insensitively before a name is added to the list — the
 * backend itself also rejects duplicates (Phase 8's input validation),
 * so this is a UX convenience (catch the mistake before a round trip),
 * not the only line of defense.
 *
 * WHY RESULTS ARE GROUPED BY PAIR, NOT RE-RANKED BY THE FRONTEND
 * --------------------------------------------------------------------
 * Each pair's result is rendered with the same InteractionResult
 * component used for the single two-drug check — the SAME assessment
 * badge, the SAME evidence cards, the SAME safety rules. This
 * component does not sort, filter, or prioritize pairs by any
 * frontend-invented notion of severity — pairs are shown in the order
 * the backend returned them, exactly matching the explicit requirement
 * to prioritize by the backend's own evidence status, not an
 * arbitrary frontend assumption.
 */
export default function MultiDrugChecker() {
  const [pending, setPending] = useState("");
  const [drugs, setDrugs] = useState([]);
  const [duplicateWarning, setDuplicateWarning] = useState(null);
  const { status, data, error, checkMultiple } = useMultiInteractionCheck();

  function addDrug() {
    const name = pending.trim();
    if (!name) return;

    const isDuplicate = drugs.some((d) => d.toLowerCase() === name.toLowerCase());
    if (isDuplicate) {
      setDuplicateWarning(`"${name}" is already in the list.`);
      return;
    }

    setDuplicateWarning(null);
    setDrugs((prev) => [...prev, name]);
    setPending("");
  }

  function removeDrug(name) {
    setDrugs((prev) => prev.filter((d) => d !== name));
  }

  function handleAddKeyDown(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      addDrug();
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    if (drugs.length < 2) return;
    checkMultiple(drugs);
  }

  const pairCount = drugs.length >= 2 ? (drugs.length * (drugs.length - 1)) / 2 : 0;

  return (
    <section className="checker-section" aria-labelledby="multi-drug-heading">
      <h2 id="multi-drug-heading">Check Multiple Medications</h2>
      <p className="section-hint">Add two or more medications, then check every combination at once.</p>

      <div className="multi-drug-add">
        <DrugInput
          id="multi-drug-add-input"
          label="Add a medication"
          value={pending}
          onChange={(v) => {
            setPending(v);
            setDuplicateWarning(null);
          }}
          onKeyDown={handleAddKeyDown}
          placeholder="e.g. aspirin"
        />
        <button type="button" onClick={addDrug} className="btn-secondary">
          Add
        </button>
      </div>

      {duplicateWarning && (
        <p className="duplicate-warning" role="alert">
          {duplicateWarning}
        </p>
      )}

      {drugs.length > 0 && (
        <ul className="drug-chip-list" aria-label="Medications to check">
          {drugs.map((name) => (
            <li key={name} className="drug-chip">
              {name}
              <button type="button" onClick={() => removeDrug(name)} aria-label={`Remove ${name}`}>
                &times;
              </button>
            </li>
          ))}
        </ul>
      )}

      <form onSubmit={handleSubmit}>
        <button type="submit" disabled={drugs.length < 2 || status === "loading"} className="btn-primary">
          {status === "loading" ? "Checking all pairs…" : `Check ${pairCount} pair${pairCount === 1 ? "" : "s"}`}
        </button>
      </form>

      {status === "loading" && <LoadingState label="Checking evidence for all medication pairs…" />}
      {status === "error" && <ErrorMessage message={error && error.message} onRetry={() => checkMultiple(drugs)} />}
      {status === "success" && data && (
        <div className="multi-results">
          {data.pairs.map((pair) => (
            <div key={`${pair.drug_a}-${pair.drug_b}`} className="pair-result">
              <h3>
                {pair.drug_a} + {pair.drug_b}
              </h3>
              <InteractionResult result={pair.result} />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
