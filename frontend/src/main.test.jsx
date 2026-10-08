import { describe, expect, it } from "vitest";
import { formatNumber, formatPercent } from "./format.js";

describe("dashboard formatting", () => {
  it("formats percentages and numbers consistently", () => {
    expect(formatPercent(0.875)).toBe("87.5%");
    expect(formatNumber(1.23456)).toBe("1.23");
  });

  it("uses safe empty values", () => {
    expect(formatPercent(undefined)).toBe("0.0%");
    expect(formatNumber(Number.NaN)).toBe("—");
  });
});
