import { useCallback, useState } from "react";
import { checkInteraction, ApiError } from "../services/api";

/**
 * Manages the request lifecycle for POST /api/interaction/check.
 * status: "idle" | "loading" | "success" | "error"
 */
export function useInteractionCheck() {
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const check = useCallback(async (drugA, drugB) => {
    setStatus("loading");
    setError(null);
    try {
      const data = await checkInteraction(drugA, drugB);
      setResult(data);
      setStatus("success");
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError("Something went wrong. Please try again."));
      setStatus("error");
    }
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setResult(null);
    setError(null);
  }, []);

  return { status, result, error, check, reset };
}
