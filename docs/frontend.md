# Frontend — Phase 9

## React architecture

A single-page app (no router — there's exactly one page) built from
small, single-purpose components composed in `pages/Home.jsx`. State is
kept as close as possible to where it's used:

- `Home.jsx` holds only the two-drug form's own state (`drugA`, `drugB`)
  plus the `useInteractionCheck` hook's request-lifecycle state.
- `MultiDrugChecker.jsx` manages its own drug list entirely internally
  — `Home.jsx` doesn't know or care how many drugs are in that list.
- Presentation components (`EvidenceCard`, `InteractionResult`,
  `SafetyNotice`, `LoadingState`, `ErrorMessage`) hold no state of their
  own beyond `EvidenceCard`'s local expand/collapse toggle — they render
  exactly what they're given.

No global state library (Redux, Zustand, Context) was needed or added —
the state that exists is genuinely local to one form or one list, and
introducing a global store would have been complexity without a
corresponding need.

## Component structure

| Component | Responsibility |
|---|---|
| `DrugInput` | Plain labeled `<input>` — no search logic. Reused by both `DrugSearch` and `MultiDrugChecker`'s "add a medication" field, so the label/input markup exists in exactly one place. |
| `DrugSearch` | Wraps `DrugInput` with autocomplete: debounced search via `useDrugSearch`, ARIA combobox pattern, full keyboard navigation. |
| `InteractionResult` | Renders one `ExplanationResponse` — the assessment badge, prose sections (only in `llm_grounded` mode), severity (with the "not stated" fallback), limitations, and the evidence list. |
| `EvidenceCard` | One expandable evidence item — id, classification, drug, and (expanded) section, manufacturer, full text, source link. |
| `MultiDrugChecker` | The "Check Multiple Medications" flow: add/remove drugs, duplicate prevention, submit, render one `InteractionResult` per returned pair. |
| `SafetyNotice` | The standing disclaimer, shown once per page load, always visible. |
| `LoadingState` / `ErrorMessage` | Shared, consistent loading and error UI used by both checkers. |

## API communication

`services/api.js` is the **only** file that calls `fetch`. Every
component goes through its four exported functions
(`searchDrugs`, `checkInteraction`, `checkMultipleInteractions`,
`checkHealth`). This one-file rule means:

- The backend's exact request/response shapes are known in one place.
- Every possible failure — a network error, a non-2xx response, a
  malformed (non-JSON) response — is normalized into the same
  `ApiError` class, so every component that calls these functions has
  exactly one error type to catch, with a `.message` that's always
  safe to show a user directly and a `.code` for any conditional
  behavior (e.g. distinguishing a rate-limit message).

## State management for async requests

Four small hooks, deliberately not one big "do everything" hook:
`useDebouncedValue` (generic), `useDrugSearch` (search-specific, built
on the debounce hook), `useInteractionCheck` and
`useMultiInteractionCheck` (one per backend endpoint). Each exposes the
same shape — a `status` string (`idle`/`loading`/`success`/`error`)
plus the relevant data/error — so components render loading/error/
success states identically regardless of which hook they're using.

## Loading and error states

Every async action has a dedicated, visible state:
- **Searching a drug name:** the suggestion dropdown shows "Searching…"
  inline, never a blocked input.
- **Checking an interaction (single or multi):** `LoadingState`'s
  spinner + `role="status"`/`aria-live="polite"`, and the submit
  button's label changes to reflect what's happening ("Checking…").
- **Backend unavailable / any other failure:** `ErrorMessage` with a
  plain-language message (never a raw exception or stack trace) and a
  "Try again" button that re-issues the exact same request.

## Evidence presentation

`InteractionResult` renders the backend's `cited_evidence` array as a
list of `EvidenceCard`s — collapsed by default so a result with several
evidence items doesn't overwhelm the page, but every field (id, source
section, manufacturer, full original text, source URL) is one click
away. **Nothing here is computed by the frontend**: the classification
label shown (e.g. "Pair-specific evidence") comes from
`utils/evidence.js`'s lookup table, which only maps the backend's own
fixed strings to readable labels — it never re-derives or upgrades a
classification.

## Safety UX

Three deliberate choices, matching the backend's own safety rules:

1. **`insufficient_evidence` is never styled or worded like "no
   interaction."** `utils/evidence.js` gives it a neutral tone and an
   explicit description ("This does not mean the medications are safe
   to combine.") — verified directly by
   `InteractionResult.test.jsx`'s test asserting the phrase never
   appears as an affirmative claim.
2. **Evidence-only fallback never looks broken.** When `mode ===
   "evidence_only"` (Gemini failed, or its response failed grounding
   validation), the UI shows a plain explanation of why, plus the full
   retrieved evidence — never a blank section where prose would have
   been.
3. **Severity always renders something.** A `null` severity renders as
   the literal string "Not stated in retrieved source" — never a blank
   line that could be mistaken for a UI bug.

## Accessibility

- Every input has a real `<label htmlFor>` association (verified in
  tests via `getByLabelText`, which only passes with correct
  label/input wiring).
- `DrugSearch` implements the ARIA combobox pattern in full: `role="combobox"`,
  `aria-expanded`, `aria-controls`, `role="listbox"`/`role="option"`,
  and complete keyboard support (Arrow Up/Down, Enter, Escape) — not
  mouse-only.
- `LoadingState` uses `role="status"` + `aria-live="polite"`;
  `ErrorMessage` uses `role="alert"` — screen readers are informed of
  state changes without the user needing to hunt for them.
- `EvidenceCard`'s expand/collapse button uses `aria-expanded` +
  `aria-controls` pointing at the panel it reveals.
- Color is never the only signal — every assessment badge also carries
  a text label, and tone colors were chosen for contrast against their
  backgrounds.
- Layout is responsive via CSS Grid with a mobile breakpoint (the
  two-drug form stacks vertically under ~620px).

## What was actually verified, and how

- **54 frontend unit/integration tests, all passing**, using React
  Testing Library against real component render output with mocked
  `services/api.js` — this proves component behavior and composition
  correctly, but does not involve a real browser or the real backend.
- **A genuine, live integration check**: the real FastAPI server was
  started in this project's build sandbox (real SQLite, real ChromaDB),
  and hit with real `curl` requests shaped exactly like the frontend's
  `api.js` sends them — confirming the actual JSON contract matches
  what the components expect. This is not the same as browser testing
  (see `docs/how-it-was-built.md` for exactly what this did and didn't
  prove, including an honest account of the real embedding model
  failing for lack of network access, and the system correctly falling
  back to a safe `retrieval_error` response rather than crashing).
- **No real browser was launched in this sandbox.** Visual rendering,
  actual click-through interaction, and the two dev servers running
  side-by-side have NOT been observed by Claude — see "Browser/local
  integration" in `docs/how-it-was-built.md` for the exact instructions
  to verify this yourself.

## Viva preparation

**"Why React?"** Component-based structure matches this UI's natural
shape (a form, a result panel, repeatable evidence cards) well; a huge
ecosystem and hooks-based state management fit a small, focused
prototype without needing a heavier framework.

**"Why separate frontend and backend?"** Clean separation of concerns —
the backend owns all clinical/evidence logic and can be tested,
versioned, and deployed independently of any UI; the frontend is a thin
presentation layer that could be swapped (a different frontend, a
mobile app, a CLI) without touching the actual decision-support logic
at all.

**"How does React communicate with FastAPI?"** Plain HTTP `fetch` calls
to FastAPI's JSON REST endpoints — see `services/api.js`. No
WebSockets, no GraphQL; the request/response cycle is simple enough
that REST is the right tool.

**"What is a REST API?"** An architectural style where resources
(drugs, interaction checks) are addressed by URLs, and standard HTTP
verbs (GET to read, POST to create/submit) describe the action —
`GET /api/drugs/search`, `POST /api/interaction/check`.

**"How does the frontend display RAG evidence?"** Each retrieved
evidence chunk (Phase 5-6's output, resolved to full detail by Phase 7)
becomes one `EvidenceCard`, showing exactly where an answer came from —
never summarized or altered by the frontend.

**"How are API errors handled?"** Every failure — network, validation,
rate limit, server error, malformed response — is normalized by
`services/api.js` into one `ApiError` type with a safe `.message`,
caught by the relevant hook, and rendered via the shared `ErrorMessage`
component with a retry option. No raw exception or stack trace ever
reaches the rendered page.

**"Why shouldn't the frontend calculate interaction severity?"**
Severity determination requires access to the actual retrieved evidence
text and the backend's grounding validation (Phase 7) — the frontend
never sees enough context to do this safely, and duplicating that logic
client-side would create two sources of truth that could disagree. The
frontend only ever displays what the backend already validated.

**"Why does the frontend display source evidence?"** So a user (or a
viva panel) can verify exactly where a claim came from — the whole
point of a RAG-grounded system is traceability, and hiding the evidence
behind a purely conversational answer would throw that away.

**"How does the UI handle insufficient evidence?"** It's shown with a
neutral (not red, not green) badge and an explicit description that
this is not the same as "no interaction" — see "Safety UX" above.
