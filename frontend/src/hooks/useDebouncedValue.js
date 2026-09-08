import { useEffect, useState } from "react";

/**
 * Returns `value`, but only updates after `delayMs` has passed without
 * a further change — used to avoid firing a search request on every
 * single keystroke.
 */
export function useDebouncedValue(value, delayMs = 300) {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);

  return debounced;
}
