"""Render a DataFrame as a Markdown table (no extra dependency needed)."""
import pandas as pd


def to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    frame = df.reset_index() if index else df
    header = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"
