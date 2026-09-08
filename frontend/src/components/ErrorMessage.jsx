/**
 * ErrorMessage.jsx
 * ===================
 * Displays a user-friendly error message — always `error.message` from
 * an ApiError (see services/api.js), NEVER a raw exception, stack
 * trace, or backend internal detail. `role="alert"` announces it
 * immediately to assistive technology, since an error is something the
 * user needs to know about right away.
 */
export default function ErrorMessage({ message, onRetry }) {
  return (
    <div className="error-message" role="alert">
      <p>{message || "Something went wrong. Please try again."}</p>
      {onRetry && (
        <button type="button" onClick={onRetry} className="btn-secondary">
          Try again
        </button>
      )}
    </div>
  );
}
