"""Shared colours and axis styling for every figure in the project."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CLASS_ORDER

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

# Congestion levels are ordered, so they use one blue ramp from light (Low)
# to dark (Severe) everywhere, rather than unrelated hues.
LEVEL_COLORS = dict(zip(CLASS_ORDER, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]))

# Categorical slots, used in this fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "legend.frameon": False,
            "font.size": 10,
        }
    )
