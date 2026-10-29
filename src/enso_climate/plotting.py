"""Figure style: the publication style of the paper figures and the helpers of the report figures.

Publication figures (scripts/make_paper_figures.py) call ``apply()`` and are drawn at their
printed size (``DOUBLE`` or ``SINGLE`` column width, in inches), so the font sizes below are the
sizes on the page: 8 pt text and 7.5 pt tick labels, inside the 6-9 pt range that AMS, AGU and
Springer journals ask for. They are saved as 300 dpi PNGs; if a PDF is ever needed, TrueType
fonts are embedded (``pdf.fonttype`` 42). Colour encodes the model family and is limited to three
hues that stay distinguishable for colour-blind readers in every pairing (checked with the dataviz
palette validator). Dark grey marks reference baselines. Variants within a family differ by line
style and marker, so no figure depends on colour alone.

    from enso_climate import plotting
    plotting.apply()
    fig, ax = plt.subplots(figsize=(plotting.DOUBLE, 3.0))

The experiment figures in reports/figures/ keep their own look (``style``, ``suptitle``,
``plot_acc_rmse`` and the further colours below; tests/test_plotting.py pins the values) and do
not call ``apply()``. Importing this module does not select a matplotlib backend; scripts call
``matplotlib.use("Agg")`` first.
"""

from __future__ import annotations

from cycler import cycler

# Neutrals: text, secondary text (ticks, legends), axis lines.
INK, INK2, AXIS = "#0b0b0b", "#52514e", "#c3c2b7"
# Model families: oscillator / warm water volume, index network / Nino-3.4, CNN; grey for baselines.
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
# Diverging pair for ENSO phase, with a grey midpoint for neutral months.
WARM, COOL, PALE = "#e34948", "#2a78d6", "#b9b8b0"
# Light fill for non-significance bands and row separators.
SHADE = "#e1e0d9"

SOLID, DASHED, DOTTED, DASHDOT = "-", (0, (5, 2)), (0, (1, 1.3)), (0, (5, 1.5, 1, 1.5))

# Figure widths in inches: AMS column (19 pc) and page (39 pc) widths, close to AGU and Springer.
SINGLE, DOUBLE = 3.2, 6.5

RC = {
    # text
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.unicode_minus": True,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    # axes: left and bottom spines only, no gridlines
    "axes.linewidth": 0.6,
    "axes.edgecolor": AXIS,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(color=[BLUE, ORANGE, AQUA, NEUTRAL]),
    # ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    # lines and markers
    "lines.linewidth": 1.5,
    "lines.markersize": 3.2,
    "lines.markeredgewidth": 0.9,
    # legend
    "legend.frameon": False,
    "legend.handlelength": 3.0,
    "legend.labelcolor": INK2,
    # output
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply() -> None:
    """Set the publication style for the rest of the session."""
    import matplotlib as mpl

    mpl.rcParams.update(RC)


def panel_label(ax, letter: str) -> None:
    """Bold (a), (b), ... label above the top-left corner of an axes."""
    ax.text(0.0, 1.03, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")


# Report figures: further neutrals (gridlines, grey series and tick labels) and series colours.
GREEN, GRID, MUTED = AQUA, SHADE, "#898781"
AMBER, PINK = "#eda100", "#e87ba4"
LIGHT_BLUE, DARK_BLUE, LIGHT_GREEN, RED = "#86b6ef", "#1c5cab", "#8fd3b6", "#b2452f"
SERIES = [BLUE, ORANGE, GREEN, AMBER]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def style(ax, title):
    """Axes style of the report figures: left-aligned title, light grid, left and bottom spines."""
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def suptitle(fig, text: str) -> None:
    """Left-aligned figure title."""
    fig.suptitle(text, x=0.01, ha="left", fontsize=10, color=INK)


def plot_acc_rmse(skill, lines: dict, path, title: str, key: str = "label", figsize=(11, 3.8),
                  xticks=range(1, 13), markersize: float = 3.5) -> None:
    """Anomaly correlation and RMSE against lead, one line per entry of ``lines``.

    ``skill`` is a skill_table with columns ``lead``, ``acc``, ``rmse`` and ``key``; ``lines`` maps a
    value of ``key`` to (display name, colour, linestyle). Solid lines are thicker and carry markers."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for value, (name, colour, dash) in lines.items():
        part = skill[skill[key] == value]
        for ax, metric in zip(axes, ("acc", "rmse")):
            ax.plot(part.lead, part[metric], color=colour, linewidth=2 if dash == "-" else 1.5, linestyle=dash,
                    marker="o" if dash == "-" else None, markersize=markersize, label=name)
    axes[0].axhline(0.5, color=AXIS, linewidth=0.8)
    style(axes[0], "Anomaly correlation")
    style(axes[1], "RMSE (°C)")
    for ax in axes:
        ax.set_xlabel("Lead (months)", color=INK2, fontsize=9)
        ax.set_xticks(xticks)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    suptitle(fig, title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
