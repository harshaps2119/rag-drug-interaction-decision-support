/**
 * services/api.js
 * ==================
 * The ONLY place in this frontend that knows how to talk to the FastAPI
 * backend. Every component goes through the functions here — none of
 * them call `fetch` directly. This keeps API-shape knowledge (URLs,
 * error envelope format, request/response fields) in one place, and is
 * exactly what makes these functions mockable in tests without any
 * component needing to know it's being tested against a fake.
 *
 * WHY THE BASE URL IS AN ENVIRONMENT VARIABLE, NOT HARD-CODED
 * ------------------------------------------------------------------
 * `VITE_API_BASE_URL` (see `.env.example`) is read via Vite's
 * `import.meta.env` at build time. This means the exact same built
 * frontend can point at a local backend during development and a
 * different URL in any other deployment, without editing source code.
 * Never commit a real `.env` file — only `.env.example` documents the
 * variable's name and a safe local-development default.
 *
 * WHY EVERY ERROR IS NORMALIZED INTO ApiError
 * ------------------------------------------------
 * The backend's error envelope is always
 * `{"error": {"code", "message", "request_id"}}` (see
 * ../../../backend/docs/api.md). Components should never need to know
 * that shape — they just catch `ApiError` and read `.message` (always
 * safe to show a user) and `.code` (for conditional UI behavior, e.g.
 * a distinct message for rate limiting). A network failure (backend
 * unreachable entirely) is normalized into the same ApiError shape
 * with code "network_error", so calling code has exactly ONE error
 * type to handle, not two.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  constructor(message, { code = "unknown_error", status = null, requestId = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.requestId = requestId;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json", ...options.headers },
      ...options,
    });
  } catch {
    // fetch itself throws for network failures (backend unreachable, DNS, etc.) —
    // never a JS error with a stack trace visible in the UI, always this.
    throw new ApiError(
      "Unable to reach the server. Please check that the backend is running and try again.",
      { code: "network_error" }
    );
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    // Response wasn't valid JSON at all — treated as malformed, not silently ignored.
  }

  if (!response.ok) {
    const errInfo = body && body.error ? body.error : null;
    const fallbackMessages = {
      429: "Too many requests. Please wait a moment and try again.",
      413: "That request was too large.",
      500: "Something went wrong on the server. Please try again.",
    };
    throw new ApiError(
      (errInfo && errInfo.message) || fallbackMessages[response.status] || "Something went wrong. Please try again.",
      {
        code: (errInfo && errInfo.code) || "http_error",
        status: response.status,
        requestId: errInfo && errInfo.request_id,
      }
    );
  }

  if (body === null) {
    throw new ApiError("The server returned an unexpected response. Please try again.", {
      code: "malformed_response",
      status: response.status,
    });
  }

  return body;
}

/** GET /api/drugs/search?q=... — returns [] for a short/empty query without calling the backend at all. */
export async function searchDrugs(query) {
  const trimmed = (query || "").trim();
  if (trimmed.length < 2) return [];
  const params = new URLSearchParams({ q: trimmed });
  return request(`/api/drugs/search?${params.toString()}`);
}

/** POST /api/interaction/check */
export async function checkInteraction(drugA, drugB) {
  return request("/api/interaction/check", {
    method: "POST",
    body: JSON.stringify({ drug_a: drugA, drug_b: drugB }),
  });
}

/** POST /api/interaction/check-multiple */
export async function checkMultipleInteractions(drugs) {
  return request("/api/interaction/check-multiple", {
    method: "POST",
    body: JSON.stringify({ drugs }),
  });
}

/** GET /api/health */
export async function checkHealth() {
  return request("/api/health");
}
