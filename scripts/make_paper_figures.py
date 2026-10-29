"""Publication versions of the manuscript figures, drawn from saved results only.

No model is fitted. Reads reports/tables/*.csv (and data/processed/enso_monthly.csv for the
RONI history) and writes 300 dpi PNGs to paper/figures/, without embedded titles so that the
captions carry the description. Reference lines are drawn only where the text uses them
(ACC = 0.5, zero difference, the RONI thresholds); significance is shown by filled markers
rather than shaded bands.

Fonts, sizes, ticks, spines and colours come from enso_climate.plotting: blue = seasonal
recharge oscillator or warm water volume, orange = index network or Niño-3.4, aqua = CNN, dark
grey = reference baselines. Variants within a family differ by line style and marker.
    python scripts/make_paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from enso_climate import plotting  # noqa: E402
from enso_climate.plotting import (AQUA, AXIS, BLUE, COOL, DASHDOT, DASHED, DOTTED, INK2, NEUTRAL,  # noqa: E402
                                    ORANGE, PALE, SOLID, WARM)
from enso_climate.plotting import DOUBLE as WIDTH  # noqa: E402
from enso_climate.plotting import panel_label as panel  # noqa: E402

TABLES = ROOT / "reports" / "tables"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "paper" / "figures"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
COLUMNS = ["model", "start", "lead", "forecast", "observed"]

plotting.apply()


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, parse_dates=["start"])


def line(ax, x, y, spec: dict, label: str, filled=None) -> None:
    """spec: colour, dash and optional marker. Markers are open, and solid where ``filled`` is True."""
    marker = spec.get("marker")
    ax.plot(x, y, color=spec["colour"], linestyle=spec.get("dash", SOLID), linewidth=spec.get("width", 1.5),
            marker=marker, markerfacecolor="white" if marker else None, label=label)
    if filled is not None and marker:
        x, y, filled = np.asarray(x), np.asarray(y), np.asarray(filled, dtype=bool)
        ax.plot(x[filled], y[filled], linestyle="none", marker=marker, color=spec["colour"])


def skill_threshold(ax) -> None:
    """ACC = 0.5, the usual threshold for useful skill, referred to in the text."""
    ax.axhline(0.5, color=AXIS, linewidth=0.8, zorder=0)


def acc(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [{"model": model, "lead": lead, "acc": np.corrcoef(g.forecast, g.observed)[0, 1]}
            for (model, lead), g in frame.groupby(["model", "lead"])]
    return pd.DataFrame(rows)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def legend_below(ax, ncol: int = 2, offset: float = -0.17) -> None:
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, offset), ncol=ncol)


def roni_history() -> None:
    m = pd.read_csv(PROCESSED / "enso_monthly.csv", parse_dates=["date"]).set_index("date")
    roni, phase = m["roni"], m["enso_phase_relative"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    for name, colour, label in (("neutral", PALE, "Neutral"), ("el_nino", WARM, "El Niño episode"),
                                ("la_nina", COOL, "La Niña episode")):
        ax.fill_between(roni.index, 0, roni, where=(phase == name).to_numpy(), color=colour, linewidth=0,
                        step="mid", label=label)
    for level in (0.5, -0.5):  # episode thresholds
        ax.axhline(level, color=INK2, linewidth=0.7, linestyle=DASHED)
    ax.set_ylabel("Relative ONI (°C)")
    ax.set_xlim(pd.Timestamp("1949-07-01"), pd.Timestamp("2027-01-01"))
    legend_below(ax, ncol=3, offset=-0.12)
    save(fig, "roni_history")


def summary_forecasts() -> pd.DataFrame:
    """The strict-protocol forecasts shown in the headline figure."""
    e7, e13, e8, e3 = read("e7_forecasts.csv"), read("e13_forecasts.csv"), read("e8_forecasts.csv"), read("e3_ro_forecasts.csv")
    parts = [e7[e7.label == "cs_lim|ro"].assign(model="recharge_oscillator"),
             e7[e7.label == "ridge|ro"].assign(model="ridge_ro"),
             e13[e13.model == "ro_dmi"], e8[e8.model == "cnn"],
             e3[e3.model == "st"].assign(model="net_ro"), e3[e3.model == "mt_auto"].assign(model="net_ro_mt")]
    return pd.concat([p[COLUMNS] for p in parts], ignore_index=True)


SKILL_LINES = {  # model key: (label, line spec)
    "recharge_oscillator": ("Seasonal recharge oscillator", dict(colour=BLUE, marker="o")),
    "ro_dmi": ("Seasonal recharge oscillator + DMI", dict(colour=BLUE, dash=DASHED, width=1.2)),
    "net_ro": ("Network, single-task", dict(colour=ORANGE, marker="s")),
    "net_ro_mt": ("Network, multi-task", dict(colour=ORANGE, dash=DASHED, marker="^")),
    "cnn": ("CNN on SST and D20 maps", dict(colour=AQUA, marker="D")),
    "ridge_ro": ("Ridge, Niño-3.4 + WWV", dict(colour=NEUTRAL, dash=DASHDOT)),
}


def skill_by_lead() -> None:
    skill = acc(summary_forecasts())
    fig, ax = plt.subplots(figsize=(WIDTH, 3.2))
    for key, (label, spec) in SKILL_LINES.items():
        part = skill[skill.model == key]
        line(ax, part.lead, part.acc, spec, label)
    skill_threshold(ax)
    ax.set_xlabel("Lead (months)")
    ax.set_ylabel("Anomaly correlation")
    ax.set_xticks(range(1, 25))
    ax.set_xlim(0.5, 24.5)
    legend_below(ax)
    save(fig, "skill_by_lead")


MTL_FAMILIES = {  # file stem of the *_vs_single_task.csv comparison: row-group label
    "e3_sst_wwv": "SST + WWV",
    "e3_ro": "Niño-3.4 + WWV",
    "e3_ro_tasks": "Niño-3.4 + WWV (factorial)",
    "e3_ro_backbone": "Oscillator backbone",
}
MTL_ORDER = ["mt_w0.25", "mt_w0.5", "mt_w1", "mt_auto", "dyn_w0.5", "dyn_w1", "dyn_auto"]
FAMILY_GAP = 0.8  # extra space between row groups, in rows


def mtl_label(model: str) -> str:
    tasks, weight = model.split("_", 1)
    tasks = "derived" if tasks == "mt" else "dynamical"
    weight = "w selected" if weight == "auto" else f"w = {weight[1:]}"
    return f"{tasks}, {weight}"


def mtl_vs_single_task() -> None:
    rows = []
    for stem, family in MTL_FAMILIES.items():
        t = pd.read_csv(TABLES / f"{stem}_vs_single_task.csv")
        t = t[(t.vs == "st") & (t.metric == "acc") & t.model.isin(MTL_ORDER)]
        for model in [m for m in MTL_ORDER if m in set(t.model)]:
            rows.append(t[t.model == model].assign(family=family, config=mtl_label(model)))
    table = pd.concat(rows, ignore_index=True)
    configs = table[["family", "config"]].drop_duplicates().reset_index(drop=True)
    group = {family: i for i, family in enumerate(configs.family.unique())}
    position = [i + FAMILY_GAP * group[r.family] for i, r in enumerate(configs.itertuples(index=False))]
    top = max(position)
    y = {tuple(r): top - p for r, p in zip(configs.itertuples(index=False), position)}
    leads = (3, 6, 9, 12)
    fig, axes = plt.subplots(1, len(leads), figsize=(WIDTH, 4.6), sharey=True)
    for ax, lead, letter in zip(axes, leads, "abcd"):
        part = table[table.lead == lead]
        for row in part.itertuples(index=False):
            yy = y[(row.family, row.config)]
            derived = row.config.startswith("derived")
            colour = COOL if derived else WARM
            ax.plot([row.ci_low, row.ci_high], [yy, yy], color=colour, linewidth=1.1)
            ax.plot(row.difference, yy, "o" if derived else "s", markersize=3.6, color=colour,
                    markerfacecolor=colour if row.significant else "white")
        ax.axvline(0, color=INK2, linewidth=0.8, zorder=0)  # no difference from the single-task twin
        ax.set_title(f"({letter}) {lead}-month lead")
        ax.set_xlabel("ACC difference")
        ax.set_xlim(-0.12, 0.12)
        ax.set_xticks([-0.1, 0, 0.1])
    labels = [f"{r.family}, {r.config}" for r in configs.itertuples(index=False)]
    axes[0].set_yticks([y[tuple(r)] for r in configs.itertuples(index=False)], labels, fontsize=7)
    axes[0].set_ylim(-0.7, top + 0.7)
    save(fig, "mtl_vs_single_task")


def oscillator_operators() -> None:
    table = pd.read_csv(TABLES / "ro_operator.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    # colour follows the variable the term starts from: orange = Niño-3.4, blue = warm water volume
    groups = [(axes[0], {"nino34 self-coupling": ("Niño-3.4", dict(colour=ORANGE, marker="o")),
                         "WWV self-coupling": ("Warm water volume", dict(colour=BLUE, marker="s"))},
               "Self-coupling coefficient"),
              (axes[1], {"WWV -> nino34": ("Recharge, WWV to Niño-3.4", dict(colour=BLUE, marker="s")),
                         "nino34 -> WWV": ("Discharge, Niño-3.4 to WWV", dict(colour=ORANGE, dash=DASHED, marker="o"))},
               "Coupling coefficient")]
    for (ax, terms, ylabel), letter in zip(groups, "ab"):
        for term, (label, spec) in terms.items():
            part = table[table.term == term].set_index("month").reindex(MONTHS)
            ax.fill_between(range(12), part.ci_low, part.ci_high, color=spec["colour"], alpha=0.15, linewidth=0)
            line(ax, range(12), part.value, spec, label)
        ax.set_xticks(range(12), [m[0] for m in MONTHS])
        ax.set_xlabel("Calendar month of the one-month step")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
        panel(ax, letter)
    fig.tight_layout(w_pad=2.5)
    save(fig, "oscillator_operators")


def cnn_protocols() -> None:
    f = read("e10b_forecasts.csv")
    e7 = read("e7_forecasts.csv")
    f = pd.concat([f, e7[e7.label == "cs_lim|ro"].assign(model="recharge_oscillator")[f.columns]], ignore_index=True)
    skill = acc(f)
    lines = {"cnn": ("CNN, strict protocol", dict(colour=AQUA, marker="D")),
             "cnn|P2": ("CNN, epochs chosen on test years", dict(colour=AQUA, dash=DASHED, width=1.7)),
             "cnn|P4": ("CNN, random monthly split", dict(colour=AQUA, dash=DOTTED, width=1.9)),
             "recharge_oscillator": ("Seasonal recharge oscillator, strict protocol", dict(colour=BLUE, marker="o"))}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    for key, (label, spec) in lines.items():
        part = skill[skill.model == key]
        line(ax, part.lead, part.acc, spec, label)
    skill_threshold(ax)
    ax.set_xlabel("Lead (months)")
    ax.set_ylabel("Anomaly correlation")
    ax.set_xticks(range(1, 25))
    ax.set_xlim(0.5, 24.5)
    ax.set_ylim(-0.15, 1.02)
    legend_below(ax)
    save(fig, "cnn_protocols")


SHORTCUT_LINES = {  # colour = model, line style = shortcut
    "st|P2": ("Network, epochs chosen on test years", dict(colour=ORANGE, dash=DASHED, marker="o")),
    "st|P4": ("Network, random monthly split", dict(colour=ORANGE, marker="o")),
    "st|P6": ("Network, centred (ONI-style) inputs", dict(colour=ORANGE, dash=DASHDOT, marker="D")),
    "st|P1": ("Network, full-record preprocessing", dict(colour=ORANGE, dash=DOTTED, marker="o")),
    "ridge_ro|P4": ("Ridge, random monthly split", dict(colour=NEUTRAL, marker="s")),
    "ridge_ro|P1": ("Ridge, full-record preprocessing", dict(colour=NEUTRAL, dash=DOTTED, marker="s")),
    "recharge_oscillator|P1": ("Oscillator, full-record preprocessing", dict(colour=BLUE, dash=DOTTED, marker="^")),
}


def shortcut_inflation() -> None:
    table = pd.read_csv(TABLES / "e10_inflation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True)
    for ax, period, letter in zip(axes, ("1984-2026", "2000-2023"), "ab"):
        for key, (label, spec) in SHORTCUT_LINES.items():
            part = table[(table.period == period) & (table.shortcut == key)]
            line(ax, part.lead, part.difference, spec, label)
        ax.set_title(f"({letter}) Test years " + period.replace("-", "–"))
        ax.set_xlabel("Lead (months)")
        ax.set_xticks(range(1, 13))
    axes[0].set_ylabel("ACC gain over strict protocol")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "shortcut_inflation")


FOUR = [dict(colour=BLUE, marker="o"), dict(colour=ORANGE, marker="s"), dict(colour=AQUA, marker="^"),
        dict(colour=NEUTRAL, dash=DASHED, marker="D")]
LEAD_LAG_GROUPS = {
    "SST (ERSST v6)": {"wtio_ssta": "WTIO", "etio_ssta": "ETIO", "dmi": "DMI"},
    "Zonal wind (CCMP)": {"u_wpac_anom": "West Pacific", "u_cpac_anom": "Central Pacific", "u_epac_anom": "East Pacific",
                          "u_eqio_anom": "Equatorial Indian"},
    "Precipitation (ERA5)": {"pr_nino4_anom": "Niño-4", "pr_nino34_anom": "Niño-3.4", "pr_wtio_anom": "WTIO",
                             "pr_etio_anom": "ETIO"},
}


def significant(part: pd.DataFrame, metric: str) -> np.ndarray:
    """95% significance for each lead, using that series' effective-sample-size threshold."""
    return (part[metric].abs() >= part.r_95pct_threshold).to_numpy()


def lead_lag_monthly() -> None:
    table = pd.read_csv(TABLES / "lead_lag_monthly.csv")
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.6), sharex=True, sharey="row")
    letters = iter("abcdef")
    for row, metric in enumerate(("r", "partial_r_given_target_now")):
        for col, (group, names) in enumerate(LEAD_LAG_GROUPS.items()):
            ax = axes[row, col]
            for spec, (column, label) in zip(FOUR, names.items()):
                part = table[table.predictor == column].dropna(subset=[metric])
                if part.empty:
                    continue
                line(ax, part.lead, part[metric], spec, label, filled=significant(part, metric))
            letter = next(letters)
            if row == 0:
                ax.set_title(f"({letter}) {group}")
            else:
                panel(ax, letter)
                ax.set_xlabel("Lead (months)")
                handles, labels = axes[0, col].get_legend_handles_labels()
                ax.legend(handles, labels, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
    axes[0, 0].set_ylabel("Correlation")
    axes[1, 0].set_ylabel("Partial correlation")
    fig.tight_layout(h_pad=1.2)
    save(fig, "lead_lag_monthly")


def lead_lag_ohc() -> None:
    table = pd.read_csv(TABLES / "lead_lag_quarterly.csv")
    names = {"ohc700_eqpac": "0–700 m, equatorial Pacific", "ohc700_nino34": "0–700 m, Niño-3.4 box",
             "ohc2000_nino34": "0–2000 m, Niño-3.4 box"}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), sharey=True)
    for ax, metric, letter in zip(axes, ("r", "partial_r_given_target_now"), "ab"):
        for spec, (column, label) in zip(FOUR, names.items()):
            part = table[table.predictor == column].dropna(subset=[metric])
            line(ax, part.lead, part[metric], spec, f"{label} (n = {part.n.max()})", filled=significant(part, metric))
        ax.set_xticks(range(0, 5))
        ax.set_xlabel("Lead (quarters)")
        panel(ax, letter)
    axes[0].set_ylabel("Correlation")
    axes[1].set_ylabel("Partial correlation")
    axes[1].tick_params(labelleft=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=2)
    save(fig, "lead_lag_ohc")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for draw in (roni_history, skill_by_lead, mtl_vs_single_task, oscillator_operators, cnn_protocols,
                 shortcut_inflation, lead_lag_monthly, lead_lag_ohc):
        draw()
        print(f"wrote {draw.__name__}")


if __name__ == "__main__":
    main()
