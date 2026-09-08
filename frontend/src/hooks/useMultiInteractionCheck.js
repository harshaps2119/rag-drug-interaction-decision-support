import { useCallback, useState } from "react";
import { checkMultipleInteractions, ApiError } from "../services/api";

/**
 * Manages the request lifecycle for POST /api/interaction/check-multiple.
 * status: "idle" | "loading" | "success" | "error"
 */
export function useMultiInteractionCheck() {
  const [status, setStatus] = useState("idle");
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const checkMultiple = useCallback(async (drugs) => {
    setStatus("loading");
    setError(null);
    try {
      const result = await checkMultipleInteractions(drugs);
      setData(result);
      setStatus("success");
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError("Something went wrong. Please try again."));
      setStatus("error");
    }
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setData(null);
    setError(null);
  }, []);

  return { status, data, error, checkMultiple, reset };
}
