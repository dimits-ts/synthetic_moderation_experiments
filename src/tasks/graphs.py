
# Intervention Detection in Discussions
# Copyright (C) 2026 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


MARKERS = ["o", "s", "D", "^", "v", "P", "X"]
HATCHES = ["//", "\\\\", "xx", "oo", "..", "**", "++", "--"]

COLORBLIND_PALETTE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999",  # mid gray (neutral, high legibility)
    "#8DD3C7",  # light teal (tone-safe extension)
    "#FDB462",  # light orange (paired but distinct in luminance)
    "#B3DE69",  # light yellow-green (safe vs. green due to brightness)
    "#80B1D3",  # soft blue (lightened blue variant)
    "#FB8072",  # soft coral (distinct from vermillion)
    "#CAB2D6",  # lavender (low-saturation purple)
    "#BC80BD",  # plum (dark purple contrast partner)
]


def seaborn_setup() -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        font="serif",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        },
    )

    plt.rcParams.update(
        {
            "text.usetex": False,
            # Figure
            "figure.figsize": (12, 8),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Fonts
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Nimbus Roman"],
            "font.size": 18,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 22,
            "figure.titlesize": 22,
            "figure.labelsize": 22,
            # Axes
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "axes.grid": False,
            # Ticks
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            # Lines
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            # Legend
            "legend.frameon": False,
            "legend.loc": "best",
            # Math text
            "mathtext.fontset": "cm",
            # PDF/PS output (important for LaTeX + journals)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_palette(COLORBLIND_PALETTE)


def save_plot(path: Path) -> None:
    """
    Saves a plot to the specified filepath.

    :param path: The full path (including filename)
        where the plot will be saved.
    :type path: pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Figure saved to {path.resolve()}")


def get_sorted_labels(df, col):
    labels = sorted(df[col].dropna().unique())
    return labels
