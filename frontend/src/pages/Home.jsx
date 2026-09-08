import { useState } from "react";
import DrugSearch from "../components/DrugSearch";
import InteractionResult from "../components/InteractionResult";
import LoadingState from "../components/LoadingState";
import ErrorMessage from "../components/ErrorMessage";
import SafetyNotice from "../components/SafetyNotice";
import MultiDrugChecker from "../components/MultiDrugChecker";
import { useInteractionCheck } from "../hooks/useInteractionCheck";

/**
 * pages/Home.jsx
 * =================
 * The single page of this prototype: header, safety notice, the
 * two-drug checker, and the multi-drug checker. Composes components —
 * holds only the state needed to drive the two-drug form (the
 * multi-drug checker manages its own list state internally).
 */
export default function Home() {
  const [drugA, setDrugA] = useState("");
  const [drugB, setDrugB] = useState("");
  const { status, result, error, check } = useInteractionCheck();

  const trimmedA = drugA.trim();
  const trimmedB = drugB.trim();
  const formInvalid = !trimmedA || !trimmedB;

  function handleSubmit(event) {
    event.preventDefault();
    if (formInvalid) return;
    check(trimmedA, trimmedB);
  }

  return (
    <main className="home-page">
      <header className="page-header">
        <h1>Drug Interaction Decision Support</h1>
        <p className="subtitle">
          Evidence-backed drug interaction screening with AI-assisted explanation, grounded in
          FDA drug labeling.
        </p>
      </header>

      <SafetyNotice />

      <section className="checker-section" aria-labelledby="two-drug-heading">
        <h2 id="two-drug-heading">Check Interaction</h2>
        <form onSubmit={handleSubmit} className="drug-form" noValidate>
          <DrugSearch label="Drug 1" value={drugA} onChange={setDrugA} placeholder="e.g. warfarin" />
          <DrugSearch label="Drug 2" value={drugB} onChange={setDrugB} placeholder="e.g. ibuprofen" />
          <button type="submit" disabled={formInvalid || status === "loading"} className="btn-primary">
            {status === "loading" ? "Checking…" : "Check Interaction"}
          </button>
        </form>

        {status === "loading" && <LoadingState label="Checking interaction evidence…" />}
        {status === "error" && (
          <ErrorMessage message={error && error.message} onRetry={() => check(trimmedA, trimmedB)} />
        )}
        {status === "success" && result && <InteractionResult result={result} />}
      </section>

      <MultiDrugChecker />
    </main>
  );
}
