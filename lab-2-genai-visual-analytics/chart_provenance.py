"""
chart_provenance.py

Computes verifiable facts about what actually went into a chart, so the
caption can be conditioned on known data quality defects instead of
asserting a completeness it cannot support.

Design rule: every number produced here comes from pandas, never from the
LLM. The model's job is to phrase these facts, not to generate them.

Usage sketch:

    facts = build_chart_provenance(
        df, y_column="dissolved_oxygen_mgL",
        x_column="basin", aggregation="mean",
    )
    block = provenance_to_prompt_block(facts, relevant_flags=flags)
    # pass `block` into the caption prompt
"""

import pandas as pd


def build_chart_provenance(df, y_column, x_column=None, aggregation=None):
    """Return a dict of verifiable facts about the data behind one chart.

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset, before any chart-specific filtering.
    y_column : str
        The measure being aggregated (e.g. "dissolved_oxygen_mgL").
    x_column : str, optional
        The grouping column (e.g. "basin"). Omit for ungrouped charts.
    aggregation : str, optional
        The aggregation applied ("mean", "sum", "count", ...).
    """
    facts = {
        "total_rows_in_dataset": int(len(df)),
        "measure_column": y_column,
        "aggregation": aggregation,
    }

    # --- numeric coercion accounting ---------------------------------
    # errors="coerce" turns unconvertible values into NaN rather than
    # letting them silently corrupt the arithmetic.
    raw = df[y_column]
    numeric = pd.to_numeric(raw, errors="coerce")

    facts["rows_used_in_calculation"] = int(numeric.notna().sum())
    facts["rows_excluded_missing"] = int(raw.isna().sum())
    facts["rows_excluded_unparseable"] = int(
        (numeric.isna() & raw.notna()).sum()
    )

    # Surface WHAT failed to parse -- "<0.1" or "10.5 mg/L" are quality
    # findings in their own right, not just dropped rows.
    if facts["rows_excluded_unparseable"] > 0:
        bad = raw[numeric.isna() & raw.notna()].unique()
        facts["unparseable_examples"] = [str(v) for v in bad[:5]]

    # --- duplicate accounting ----------------------------------------
    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        facts["duplicate_rows_included"] = duplicates

    # --- group balance -----------------------------------------------
    if x_column is not None:
        counts = df[x_column].value_counts()
        facts["group_column"] = x_column
        facts["group_sizes"] = {str(k): int(v) for k, v in counts.items()}
        facts["smallest_group_n"] = int(counts.min())

        # Flag imbalance when the largest group is more than twice the
        # smallest -- a rough threshold, tune to taste.
        if counts.max() > 2 * counts.min():
            facts["group_imbalance"] = True

    return facts


def provenance_to_prompt_block(facts, relevant_flags=None):
    """Render provenance facts as a plain-text block for the caption prompt."""
    lines = []

    used = facts["rows_used_in_calculation"]
    total = facts["total_rows_in_dataset"]

    if used < total:
        excluded = []
        if facts["rows_excluded_missing"]:
            excluded.append(f"{facts['rows_excluded_missing']} missing")
        if facts["rows_excluded_unparseable"]:
            excluded.append(
                f"{facts['rows_excluded_unparseable']} unparseable"
            )
        lines.append(
            f"- Computed over {used} of {total} rows "
            f"({', '.join(excluded)} excluded)."
        )
    else:
        lines.append(f"- Computed over all {total} rows; none excluded.")

    if "unparseable_examples" in facts:
        examples = ", ".join(repr(v) for v in facts["unparseable_examples"])
        lines.append(f"- Values that would not convert to numeric: {examples}.")

    if "duplicate_rows_included" in facts:
        lines.append(
            f"- {facts['duplicate_rows_included']} duplicate row(s) are "
            f"included in this calculation and are double-counted."
        )

    if facts.get("group_imbalance"):
        sizes = ", ".join(
            f"{k}={v}" for k, v in sorted(
                facts["group_sizes"].items(), key=lambda kv: -kv[1]
            )
        )
        lines.append(
            f"- Group sizes are uneven ({sizes}), so comparison across "
            f"groups should be treated as indicative rather than conclusive."
        )

    if facts.get("smallest_group_n", 999) < 5:
        lines.append(
            f"- The smallest group has only {facts['smallest_group_n']} "
            f"observation(s); its value is unstable."
        )

    if relevant_flags:
        lines.append("- Dataset quality flags touching these columns:")
        for flag in relevant_flags:
            lines.append(f"    * {flag}")

    return "\n".join(lines)


def filter_flags_to_columns(flags, columns):
    """Keep only the quality flags that mention the columns in play.

    `flags` is expected to be a list of strings (or dicts with a
    "description" key). Adjust the accessor to match your app's structure.
    """
    selected = []
    for flag in flags:
        text = flag if isinstance(flag, str) else flag.get("description", "")
        if any(col in text for col in columns):
            selected.append(text)
    return selected
