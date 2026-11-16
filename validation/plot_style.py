"""Shared figure style for the validation/ scatter plots and choropleth maps.

Figures are drawn at their final printed size (7.2 in = full width, 3.5 in = one column),
so these font sizes are the real sizes in the paper. Same sizes as the temporal/ and
validation_od/ figures (axis labels and titles 10 pt, ticks 9 pt, legend 8 pt).
"""
import matplotlib.pyplot as plt

FONT_SIZE = 9        # base size and tick labels
LABEL_SIZE = 10      # axis labels and panel titles
LEGEND_SIZE = 8
SMALL_SIZE = 7       # text inside the plot area: statistics boxes, point labels
PANEL_LETTER_SIZE = 12

FULL_WIDTH = 7.2     # inches
COLUMN_WIDTH = 3.5

POINT_KW = dict(alpha=0.7, color="lightblue", edgecolor="k", s=30)
FIT_KW = dict(color="black", linestyle="--", linewidth=1.5)
BOX_KW = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": LABEL_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        # Math text (log-axis ticks such as 10^4, x10^3) in the same family
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "mathtext.sf": "Helvetica",
        "mathtext.cal": "Helvetica",  # default (Apple Chancery) prints a findfont warning
        "pdf.fonttype": 42,  # embed TrueType so text stays editable in the PDF
        "ps.fonttype": 42,
    })


def stats_box(ax, x, y, text, ha="left", va="top"):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, multialignment="left",
            fontsize=SMALL_SIZE, linespacing=1.15, bbox=BOX_KW)


def panel_letter(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes,
            fontsize=PANEL_LETTER_SIZE, fontweight="bold", ha="right", va="bottom")


def save(fig, stem, **kwargs):
    fig.savefig(f"{stem}.png", dpi=300, **kwargs)
    fig.savefig(f"{stem}.pdf", **kwargs)
    print(f"Saved {stem}.png and {stem}.pdf")
