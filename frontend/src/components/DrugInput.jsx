/**
 * DrugInput.jsx
 * ================
 * A plain, presentational labeled text input — no search/autocomplete
 * logic of its own. Used directly by MultiDrugChecker (a simple "add a
 * medication" field), and wrapped by DrugSearch to add autocomplete on
 * top without duplicating the label/input markup in two places.
 */
export default function DrugInput({ id, label, value, onChange, ...inputProps }) {
  return (
    <div className="drug-input">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        autoComplete="off"
        {...inputProps}
      />
    </div>
  );
}
