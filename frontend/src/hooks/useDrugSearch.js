import { useEffect, useRef, useState } from "react";
import { searchDrugs } from "../services/api";
import { useDebouncedValue } from "./useDebouncedValue";

/**
 * Debounced drug-name search against GET /api/drugs/search. A query
 * under 2 characters never reaches the network — matches the same
 * server-side minimum implicitly enforced by services/api.js.
 *
 * status: "idle" | "loading" | "success" | "error"
 */
export function useDrugSearch(query) {
  const debounced = useDebouncedValue(query, 300);
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("idle");
  const requestIdRef = useRef(0);

  useEffect(() => {
    const trimmed = debounced.trim();
    if (trimmed.length < 2) {
      setResults([]);
      setStatus("idle");
      return;
    }

    const thisRequestId = ++requestIdRef.current;
    setStatus("loading");

    searchDrugs(trimmed)
      .then((data) => {
        if (requestIdRef.current === thisRequestId) {
          setResults(data);
          setStatus("success");
        }
      })
      .catch(() => {
        if (requestIdRef.current === thisRequestId) {
          setResults([]);
          setStatus("error");
        }
      });
  }, [debounced]);

  return { results, status };
}
