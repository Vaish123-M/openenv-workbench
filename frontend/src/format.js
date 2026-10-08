export const formatPercent = (value) => `${((value || 0) * 100).toFixed(1)}%`;
export const formatNumber = (value, digits = 2) =>
  Number.isFinite(value) ? value.toFixed(digits) : "—";
