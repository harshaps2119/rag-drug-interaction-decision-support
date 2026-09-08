/**
 * LoadingState.jsx
 * ===================
 * A single, consistent loading indicator used everywhere the UI is
 * waiting on the backend (drug search, single check, multi-drug check).
 * `role="status"` + `aria-live="polite"` means screen readers announce
 * the label without interrupting whatever the user is doing.
 */
export default function LoadingState({ label = "Loading…" }) {
  return (
    <div className="loading-state" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}
