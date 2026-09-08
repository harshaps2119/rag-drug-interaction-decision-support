/**
 * SafetyNotice.jsx
 * ===================
 * The standing safety disclaimer, shown on every screen that displays
 * or could display an interaction result. Kept short and calm — not
 * alarming — but always visible, per the project's explicit design
 * requirement (evidence-based decision support, not a replacement for
 * clinical judgment).
 */
export default function SafetyNotice() {
  return (
    <aside className="safety-notice" role="note" aria-label="Safety disclaimer">
      <p>
        This tool provides evidence-based medication information for decision support and
        does not replace professional clinical judgment. Always verify potential interactions
        with a pharmacist or physician before making treatment decisions.
      </p>
    </aside>
  );
}
