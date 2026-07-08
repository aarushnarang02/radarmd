"""Assemble a human-readable evaluation report from the per-class metrics."""

from __future__ import annotations

import pandas as pd

from .metrics import critical_fnr_violations, mean_auroc
from .thresholds import DEFAULT_SENSITIVITY_FLOOR


def format_report(
    report: pd.DataFrame,
    sensitivity_floor: float = DEFAULT_SENSITIVITY_FLOOR,
    split_name: str = "test",
) -> str:
    """Return a markdown-ish text summary of a per-class metrics frame.

    Highlights the headline mean AUROC and whether every critical finding meets
    the false-negative-rate target (FNR <= 1 - floor).
    """
    max_fnr = 1.0 - sensitivity_floor
    lines: list[str] = []
    lines.append(f"# RadarMD evaluation ({split_name})")
    lines.append("")
    lines.append(f"Mean AUROC (macro): **{mean_auroc(report):.4f}**")
    lines.append("")

    show = report.copy()
    for col in ("auroc", "ap", "sensitivity", "specificity", "f1", "fnr", "threshold"):
        show[col] = show[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
    lines.append(show.to_string(index=False))
    lines.append("")

    violations = critical_fnr_violations(report, max_fnr)
    target = f"FNR <= {max_fnr:.2f} (sensitivity >= {sensitivity_floor:.2f}) on critical findings"
    if violations.empty:
        lines.append(f"PASS: {target}.")
    else:
        bad = ", ".join(f"{r.pathology} (FNR={r.fnr:.3f})" for r in violations.itertuples())
        lines.append(f"FAIL: {target}. Over target: {bad}.")
    return "\n".join(lines)


def meets_sensitivity_target(
    report: pd.DataFrame, sensitivity_floor: float = DEFAULT_SENSITIVITY_FLOOR
) -> bool:
    """True if no critical finding exceeds the false-negative-rate target."""
    return critical_fnr_violations(report, 1.0 - sensitivity_floor).empty
