"""Scores and figures computed from saved forecasts only (no model is fitted), test years 1984-2026.

    python scripts/evaluate.py COMMAND [options]

  e6-seasons      skill for February-May starts and before and after 2000
  e9-rules        E9 decision rules on the development or confirmation years: dev|confirm
  e10-p3          shortcut P3: best-of-N multi-task configuration chosen on test skill
  e11             ENSO-phase skill with out-of-sample amplitude recalibration
  e12             equal-weight combinations of the oscillator with each network
  e16             forecasts of major ENSO peaks at 6, 9 and 12 months
  e22             the largest multi-task gain the data do not exclude, per configuration
  e24             the separate 24-month index-network extension [--with-gridded]
  summary-figure  headline skill-by-lead figure (reports/figures/summary_skill.png)
  cnn-figure      the observation-only CNN under three protocols (reports/figures/e10b_cnn_inflation.png)
  table-s1        ACC with 95% block-bootstrap intervals for the main models (reports/tables/table_s1.csv)
  seasonal-target skill for a three-month-mean target, from the same forecasts

Reads reports/tables/ written by baselines.py, oscillator.py and networks.py. Project environment.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enso_climate.paths import ensure_output_directories  # noqa: E402

from enso_climate.forecast import (  # noqa: E402
    FORECAST_COLUMNS, _block_bootstrap_interval, anomaly_correlation, block_resample, bootstrap_difference, bootstrap_skill, mark_significant,
    phase_scores, recalibrate_amplitude, select_forecasts, skill_table,
)
from enso_climate.plotting import AXIS, BLUE, DARK_BLUE, GREEN, INK2, LIGHT_BLUE, MUTED, ORANGE, RED, style, suptitle  # noqa: E402

TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
LEADS = (3, 6, 9, 12, 18, 24)
COLUMNS = ["model", "start", "lead", "forecast", "observed"]


# ----------------------------------------------------------------------------
# e9-rules: the E9 decision rules
# ----------------------------------------------------------------------------

E9_LEADS = (3, 6, 9, 12)


def e9_rules() -> None:
    """Apply the E9 decision rules to the development or confirmation years: dev|confirm.

    Writes reports/tables/e9_<period>_rules.csv."""
    period = sys.argv[1] if len(sys.argv) > 1 else "dev"
    years = range(1984, 2006) if period == "dev" else range(2006, 2027)
    parts = []
    for tag in ("long", "short"):
        f = pd.concat([pd.read_csv(TABLES / "e9" / tag / f"year_{y}.csv", parse_dates=["start"]) for y in years],
                      ignore_index=True)
        parts.append(f.assign(model=tag + ":" + f.model)[FORECAST_COLUMNS])
    lin = pd.read_csv(TABLES / f"e9_linear_{period}_forecasts.csv", parse_dates=["start"])
    parts.append(select_forecasts(lin, "label", {"recharge_oscillator": "recharge_oscillator"}))
    d = pd.concat(parts, ignore_index=True)
    comparisons = {"H3": ("long:st", "short:st"), "H4": ("long:dyn_auto", "long:st"),
                   "H4_short": ("short:dyn_auto", "short:st"), "H5": ("long:dyn_auto", "recharge_oscillator")}
    rows = []
    for name, (a, b) in comparisons.items():
        for lead in E9_LEADS:
            r = bootstrap_difference(d, a, b, lead)
            acc_a = d[(d.model == a) & (d.lead == lead)]
            rows.append({"hypothesis": name, "model": a, "vs": b, "lead": lead,
                         "acc_model": anomaly_correlation(acc_a.forecast, acc_a.observed), **r})
    table = pd.DataFrame(rows)
    table["better"] = table.ci_low > 0
    table["worse"] = table.ci_high < 0
    table.round(4).to_csv(TABLES / f"e9_{period}_rules.csv", index=False)
    print(table.round(3).to_string(index=False))
    verdict = {}
    g = table.set_index(["hypothesis", "lead"])
    verdict["H3"] = g.loc["H3"].better.sum() >= 2 and not g.loc["H3"].worse.any()
    verdict["H4"] = (g.loc["H4"].difference >= 0).all() and g.loc["H4"].better.sum() >= 2
    verdict["H5"] = g.loc["H5"].better.sum() >= 2
    print(period, "verdicts (rule met?):", verdict)


# ----------------------------------------------------------------------------
# e10-p3: best-of-N multi-task selection on the test period
# ----------------------------------------------------------------------------

FAMILIES = ("e3_sst_wwv", "e3_ro", "e3_ro_tasks", "e3_ro_backbone")


def corr_columns(F: np.ndarray, o: np.ndarray) -> np.ndarray:
    F = F - F.mean(axis=0)
    o = o - o.mean()
    return (F * o[:, None]).sum(axis=0) / np.sqrt((F ** 2).sum(axis=0) * (o ** 2).sum())


def e10_p3_selection() -> None:
    """E10, shortcut P3: how much does choosing the best multi-task configuration on the test period
    inflate the multi-task advantage?

    Within each family of matched networks (same inputs, trunk and seeds; E3 v3 and E3b),
    the single-task twin is fixed. The honest multi-task estimate is the pre-declared
    'auto' variant (weight chosen on training years); the shortcut reports the best of
    the multi-task configurations by test ACC. Selection is redone inside every bootstrap
    resample, so the interval reflects the selection itself. Writes
    reports/tables/e10_p3_selection.csv."""
    rng = np.random.default_rng(0)
    rows = []
    for family in FAMILIES:
        f = pd.read_csv(TABLES / f"{family}_forecasts.csv", parse_dates=["start"])
        mt = sorted(m for m in f.model.unique() if m != "st" and not m.endswith("_auto"))
        autos = sorted(m for m in f.model.unique() if m.endswith("_auto"))
        for lead in range(1, 13):
            sub = f[f.lead == lead]
            wide = sub.pivot_table(index="start", columns="model", values="forecast")
            o = sub.groupby("start")["observed"].first().reindex(wide.index).to_numpy()
            years = wide.index.year.to_numpy()
            cols = ["st", *mt, *autos]
            F = wide[cols].to_numpy()

            def stats(idx):
                s = corr_columns(F[idx], o[idx])
                st, best, auto = s[0], s[1:1 + len(mt)].max(), s[1 + len(mt):].mean()
                return best - st, auto - st

            observed_best, observed_auto = stats(np.arange(len(o)))
            uniq = np.unique(years)
            draws = []
            for _ in range(2000):
                chosen = uniq[block_resample(len(uniq), 3, rng)]
                idx = np.concatenate([np.flatnonzero(years == y) for y in chosen])
                draws.append(stats(idx))
            draws = np.array(draws)
            rows.append({"family": family, "lead": lead, "n_mt_configs": len(mt),
                         "best_mt_minus_st": observed_best, "best_ci_low": np.percentile(draws[:, 0], 2.5),
                         "best_ci_high": np.percentile(draws[:, 0], 97.5),
                         "auto_mt_minus_st": observed_auto, "auto_ci_low": np.percentile(draws[:, 1], 2.5),
                         "auto_ci_high": np.percentile(draws[:, 1], 97.5),
                         "inflation": observed_best - observed_auto})
    table = pd.DataFrame(rows)
    table["best_significant"] = table.best_ci_low > 0
    table["auto_significant"] = (table.auto_ci_low > 0) | (table.auto_ci_high < 0)
    table.round(4).to_csv(TABLES / "e10_p3_selection.csv", index=False)
    show = table[table.lead.isin([3, 6, 9, 12])]
    print(show[["family", "lead", "n_mt_configs", "auto_mt_minus_st", "best_mt_minus_st", "best_ci_low", "best_ci_high",
                "best_significant", "inflation"]].round(3).to_string(index=False))
    print("leads where best-of-N MT is 'significantly' better than ST:", int(table.best_significant.sum()), "of", len(table),
          "| with the pre-declared auto variant:", int((table.auto_ci_low > 0).sum()))


# ----------------------------------------------------------------------------
# e11: ENSO-phase skill
# ----------------------------------------------------------------------------


def phase_forecasts() -> pd.DataFrame:
    e7 = pd.read_csv(TABLES / "e7_forecasts.csv", parse_dates=["start"])
    e8 = pd.read_csv(TABLES / "e8_forecasts.csv", parse_dates=["start"])
    e3 = pd.read_csv(TABLES / "e3_ro_forecasts.csv", parse_dates=["start"])
    bb = pd.read_csv(TABLES / "e3_ro_backbone_forecasts.csv", parse_dates=["start"])
    d = pd.concat([select_forecasts(e7, "label", {"cs_lim|ro": "recharge_oscillator"}),
                   select_forecasts(e3, "model", {"st": "net_ro", "mt_auto": "net_ro_multitask"}),
                   select_forecasts(bb, "model", {"st": "net_on_oscillator"}),
                   select_forecasts(e8, "model", {"cnn": "cnn"})])
    return d[d.lead <= 12].reset_index(drop=True)


def e11_phase_scores() -> None:
    """E11: ENSO phase (La Niña / neutral / El Niño) skill of the monthly forecasts.

    Forecasts and observations are classed with ±0.5 °C thresholds. Damped forecasts rarely
    cross the thresholds at long leads, so each model's forecasts are also recalibrated in
    amplitude: for test year Y the scale is std(observed)/std(forecast) over that model's
    forecasts verified in earlier test years only (at least 10), as an operational centre
    could do. Scored on 1994-2026. Writes reports/tables/e11_phase_scores.csv."""
    d = phase_forecasts()
    d["recalibrated"] = d.groupby(["model", "lead"], group_keys=False).apply(recalibrate_amplitude)
    d = d[d.origin_year >= 1994]
    rows = []
    for (model, lead), g in d.groupby(["model", "lead"]):
        for kind in ("forecast", "recalibrated"):
            rows.append({"model": model, "lead": lead, "forecasts": kind, **phase_scores(g[kind].to_numpy(), g.observed.to_numpy())})
    table = pd.DataFrame(rows)
    table.round(4).to_csv(TABLES / "e11_phase_scores.csv", index=False)
    for metric in ("heidke", "el_nino_hit_rate", "el_nino_false_alarm_ratio"):
        print(metric)
        print(table.pivot_table(index=["forecasts", "model"], columns="lead", values=metric)[[3, 6, 9, 12]].round(2).to_string())


# ----------------------------------------------------------------------------
# e12: equal-weight combinations
# ----------------------------------------------------------------------------


def e12_combination() -> None:
    """E12: equal-weight combinations of the seasonal recharge oscillator with each network (no fitted weights).

    For each start and lead the oscillator's forecast is averaged with the CNN, ConvLSTM or
    index-network forecast, and the combination's ACC is compared with the oscillator's by the
    block bootstrap. Reads the saved forecasts of E7, E8 and E3 (the index network stops at
    12 months); writes reports/tables/e12_combination.csv."""
    def read(name):
        return pd.read_csv(TABLES / name, parse_dates=["start"])

    e7, e8, e3 = read("e7_forecasts.csv"), read("e8_forecasts.csv"), read("e3_ro_forecasts.csv")
    oscillator = select_forecasts(e7, "label", {"cs_lim|ro": "RO"})
    networks = {"cnn": select_forecasts(e8, "model", {"cnn": "cnn"}),
                "convlstm": select_forecasts(e8, "model", {"convlstm": "convlstm"}),
                "net_ro": select_forecasts(e3, "model", {"st": "net_ro"})}
    rows = []
    for name, network in networks.items():
        label = f"RO + {name}"
        pairs = oscillator.merge(network, on=["start", "lead"], suffixes=("", "_network"))
        combination = pairs[["origin_year", "start", "start_month", "lead", "observed"]].assign(
            model=label, forecast=(pairs.forecast + pairs.forecast_network) / 2)
        both = pd.concat([oscillator[oscillator.start.isin(pairs.start)], combination], ignore_index=True)
        for lead in LEADS:
            part = combination[combination.lead == lead]
            if part.empty:
                continue
            rows.append({"combination": label, "lead": lead, "acc": anomaly_correlation(part.forecast, part.observed),
                         **bootstrap_difference(both, label, "RO", lead)})
    table = mark_significant(pd.DataFrame(rows))
    table.round(4).to_csv(TABLES / "e12_combination.csv", index=False)
    print(table.round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# e6-seasons: spring starts and the eras before and after 2000
# ----------------------------------------------------------------------------

SPRING = (2, 3, 4, 5)  # February-May starts


def e6_seasons() -> None:
    """E6: ACC for February-May starts against other starts, and for starts in 1984-1999 against 2000-2026.

    Uses the saved forecasts of E3, E7 and E8, leads 1-12. Writes reports/tables/e6_spring_barrier.csv
    and e6_pre_post_2000.csv (one row per model, columns by period and lead)."""
    def read(name):
        return pd.read_csv(TABLES / name, parse_dates=["start"])

    e7, e8, e3, bb = (read("e7_forecasts.csv"), read("e8_forecasts.csv"), read("e3_ro_forecasts.csv"),
                      read("e3_ro_backbone_forecasts.csv"))
    d = pd.concat([select_forecasts(e7, "label", {"cs_lim|ro": "recharge_oscillator", "ridge|ro": "ridge_ro"}),
                   select_forecasts(e3, "model", {"st": "net_ro"}),
                   select_forecasts(bb, "model", {"st": "net_on_oscillator"}),
                   select_forecasts(e8, "model", {"cnn": "cnn"})], ignore_index=True)
    d = d[d.lead <= 12]
    splits = {"e6_spring_barrier.csv": (np.where(d.start_month.isin(SPRING), "spring", "other"), ["spring", "other"]),
              "e6_pre_post_2000.csv": (np.where(d.start.dt.year < 2000, "1984-1999", "2000-2026"), ["1984-1999", "2000-2026"])}
    for name, (period, order) in splits.items():
        rows = [{"model": model, "period": p, "lead": lead, "acc": anomaly_correlation(g.forecast, g.observed)}
                for (model, p, lead), g in d.assign(period=period).groupby(["model", "period", "lead"])]
        table = pd.DataFrame(rows).pivot(index="model", columns=["period", "lead"], values="acc")
        table = table[[(p, lead) for p in order for lead in range(1, 13)]]
        table.round(3).to_csv(TABLES / name)
        print(table.loc[:, [(p, lead) for p in order for lead in (6, 9, 12)]].round(2).to_string())


# ----------------------------------------------------------------------------
# e22: equivalence bound for the multi-task null
# ----------------------------------------------------------------------------

E22_FAMILIES = {"e3_sst_wwv": "SST + WWV", "e3_ro": "Niño-3.4 + WWV", "e3_ro_tasks": "Niño-3.4 + WWV, factorial",
                "e3_ro_backbone": "oscillator backbone"}
E22_LEADS = (3, 6, 9, 12)


def e22_equivalence() -> None:
    """E22: the largest multi-task gain the data do not exclude.

    For each multi-task configuration of E3 and E3b, the mean ACC difference from its single-task
    twin over leads 3, 6, 9 and 12, with a 95% moving-block bootstrap interval over test years of
    that mean. Writes reports/tables/e22_equivalence.csv."""
    rows = []
    for stem, family in E22_FAMILIES.items():
        f = pd.read_csv(TABLES / f"{stem}_forecasts.csv", parse_dates=["start"])
        f = f[f.lead.isin(E22_LEADS)]
        st = f[f.model == "st"].set_index(["start", "lead"])
        for model in sorted(set(f.model) - {"st"}):
            mt = f[f.model == model].set_index(["start", "lead"])
            common = st.index.intersection(mt.index)
            pair = pd.DataFrame({"st": st.loc[common, "forecast"].to_numpy(), "mt": mt.loc[common, "forecast"].to_numpy(),
                                 "obs": st.loc[common, "observed"].to_numpy(), "year": st.loc[common, "origin_year"].to_numpy(),
                                 "lead": common.get_level_values("lead")})

            def mean_gain(idx, pair=pair):
                p = pair.iloc[idx]
                return np.mean([anomaly_correlation(g.mt, g.obs) - anomaly_correlation(g.st, g.obs)
                                for _, g in p.groupby("lead")])

            low, high = _block_bootstrap_interval(pair.year.to_numpy(), mean_gain, 3, 2000, 0)
            rows.append({"family": family, "model": model, "mean_difference": mean_gain(np.arange(len(pair))),
                         "ci_low": low, "ci_high": high})
    table = pd.DataFrame(rows)
    table.round(4).to_csv(TABLES / "e22_equivalence.csv", index=False)
    print(table.round(3).to_string(index=False))
    print("configurations:", len(table), "| largest upper bound:", round(table.ci_high.max(), 3),
          "| median upper bound:", round(table.ci_high.median(), 3))


# ----------------------------------------------------------------------------
# e16: major ENSO peaks
# ----------------------------------------------------------------------------

EVENTS = {"1997/98 El Niño": ("1997-09", "1998-02", "max"), "2015/16 El Niño": ("2015-09", "2016-02", "max"),
          "2023/24 El Niño": ("2023-09", "2024-02", "max"), "1988/89 La Niña": ("1988-09", "1989-02", "min"),
          "1999/2000 La Niña": ("1999-09", "2000-02", "min"), "2010/11 La Niña": ("2010-09", "2011-02", "min"),
          "2009/10 El Niño": ("2009-09", "2010-02", "max"), "1991/92 El Niño": ("1991-09", "1992-02", "max")}


def e16_events() -> None:
    """E16: forecasts of major ENSO peaks at 6, 9 and 12 months.

    For each event the target is the observed peak month of the monthly Niño-3.4 anomaly
    (training-only anomalies, as in all forecasts). Reports each model's forecast for that
    month made 6, 9 and 12 months earlier. Writes reports/tables/e16_events.csv."""
    def read(name):
        return pd.read_csv(TABLES / name, parse_dates=["start"])

    e13, e8, e3 = read("e13_forecasts.csv"), read("e8_forecasts.csv"), read("e3_ro_forecasts.csv")
    d = pd.concat([select_forecasts(e13, "model", {"ro_dmi": "oscillator+DMI"}, COLUMNS),
                   select_forecasts(e8, "model", {"cnn": "CNN"}, COLUMNS),
                   select_forecasts(e3, "model", {"st": "network"}, COLUMNS)],
                  ignore_index=True)
    d["target"] = [s + pd.DateOffset(months=int(k)) for s, k in zip(d.start, d.lead)]
    obs = d[d.lead == 1].drop_duplicates("target").set_index("target")["observed"].sort_index()
    rows = []
    for name, (lo, hi, how) in EVENTS.items():
        window = obs.loc[lo:hi]
        peak = window.idxmax() if how == "max" else window.idxmin()
        for lead in (6, 9, 12):
            for model, g in d[(d.target == peak) & (d.lead == lead)].groupby("model"):
                rows.append({"event": name, "peak_month": f"{peak:%Y-%m}", "observed": window.loc[peak], "lead": lead,
                             "model": model, "forecast": float(g.forecast.iloc[0])})
    table = pd.DataFrame(rows)
    table["fraction_of_peak"] = table.forecast / table.observed
    table.round(3).to_csv(TABLES / "e16_events.csv", index=False)
    print(table.pivot_table(index=["event", "peak_month", "observed", "lead"], columns="model", values="forecast").round(2).to_string())
    print(table.groupby(["model", "lead"]).fraction_of_peak.median().unstack().round(2).to_string())


# ----------------------------------------------------------------------------
# summary-figure and cnn-figure
# ----------------------------------------------------------------------------


SUMMARY_LINES = {  # key: (label, colour, dash)
    "ro_dmi": ("Seasonal recharge oscillator + DMI", RED, "-"),
    "recharge_oscillator": ("Seasonal recharge oscillator", ORANGE, "-"),
    "net_ro": ("Network (Niño-3.4 + WWV), single-task", GREEN, "-"),
    "net_ro_mt": ("Network, multi-task", LIGHT_BLUE, "-"),
    "cnn": ("CNN on SST + D20 maps", BLUE, "-"),
    "ridge_ro": ("Ridge (Niño-3.4 + WWV)", MUTED, "-"),
    "net_ro_random_split": ("Same network, random monthly split (leaky)", GREEN, (0, (4, 3))),
}


def summary_forecasts() -> pd.DataFrame:
    def read(name):
        return pd.read_csv(TABLES / name, parse_dates=["start"])

    e7, e13, e8, e3 = read("e7_forecasts.csv"), read("e13_forecasts.csv"), read("e8_forecasts.csv"), read("e3_ro_forecasts.csv")
    p4 = read("e10_shortcut_forecasts.csv")
    return pd.concat([select_forecasts(e7, "label", {"cs_lim|ro": "recharge_oscillator", "ridge|ro": "ridge_ro"}, COLUMNS),
                      select_forecasts(e13, "model", {"ro_dmi": "ro_dmi"}, COLUMNS),
                      select_forecasts(e8, "model", {"cnn": "cnn"}, COLUMNS),
                      select_forecasts(e3, "model", {"st": "net_ro", "mt_auto": "net_ro_mt"}, COLUMNS),
                      select_forecasts(p4, "model", {"st|P4": "net_ro_random_split"}, COLUMNS)], ignore_index=True)


def summary_figure() -> None:
    """Headline figure: strict observation-only skill vs lead, with the inflated (random-split)
    network for contrast. Uses saved forecasts from E3, E7, E8, E10 and E13.
    Writes reports/figures/summary_skill.png and reports/tables/summary_skill.csv."""
    d = summary_forecasts()
    skill = skill_table(d)[["model", "lead", "acc"]]
    skill.round(4).to_csv(TABLES / "summary_skill.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for key, (label, colour, dash) in SUMMARY_LINES.items():
        part = skill[skill.model == key]
        ax.plot(part.lead, part.acc, color=colour, linestyle=dash, linewidth=2, marker="o" if dash == "-" else None,
                markersize=3, label=label)
    ax.axhline(0.5, color=AXIS, linewidth=0.8)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    style(ax, "Anomaly correlation, monthly Niño-3.4, rolling origin 1984–2026")
    ax.set_xlabel("Lead (months)", color=INK2, fontsize=9)
    ax.set_xticks(range(1, 25))
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    suptitle(fig, "Observation-only ENSO forecasts under strict evaluation, and what a leaky split would show")
    fig.tight_layout()
    fig.savefig(FIGURES / "summary_skill.png", dpi=150)
    plt.close(fig)
    print(skill.pivot(index="model", columns="lead", values="acc")[[3, 6, 9, 12, 18, 24]].round(2).to_string())


CNN_LINES = {"cnn|P4": ("CNN, random monthly split", DARK_BLUE, (0, (4, 3))),
         "cnn|P2": ("CNN, epochs chosen on test years", BLUE, (0, (1, 2))),
         "recharge_oscillator": ("Seasonal recharge oscillator (strict)", ORANGE, "-"),
         "cnn": ("CNN (strict)", BLUE, "-")}


def cnn_figure() -> None:
    """E10b figure: the same observation-only CNN under the strict protocol and two shortcuts,
    with the seasonal recharge oscillator for reference. Writes reports/figures/e10b_cnn_inflation.png."""
    f = pd.read_csv(TABLES / "e10b_forecasts.csv", parse_dates=["start"])
    e7 = pd.read_csv(TABLES / "e7_forecasts.csv", parse_dates=["start"])
    f = pd.concat([f, select_forecasts(e7, "label", {"cs_lim|ro": "recharge_oscillator"}, list(f.columns))], ignore_index=True)
    skill = skill_table(f)[["model", "lead", "acc"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for key, (label, colour, dash) in CNN_LINES.items():
        part = skill[skill.model == key]
        ax.plot(part.lead, part.acc, color=colour, linestyle=dash, linewidth=2, label=label)
    ax.axhline(0.5, color=AXIS, linewidth=0.8)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    style(ax, "Anomaly correlation, monthly Niño-3.4, 1984–2026")
    ax.set_xlabel("Lead (months)", color=INK2, fontsize=9)
    ax.set_xticks(range(1, 25))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    suptitle(fig, "One observation-only CNN, three evaluation protocols")
    fig.tight_layout()
    fig.savefig(FIGURES / "e10b_cnn_inflation.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# table-s1: intervals for the main models
# ----------------------------------------------------------------------------

S1_NAMES = {"ro_dmi": "Seasonal recharge oscillator + DMI", "recharge_oscillator": "Seasonal recharge oscillator",
         "net_on_oscillator": "Network on the oscillator", "net_ro": "Network (single-task)",
         "net_ro_mt": "Network (multi-task)", "cnn": "CNN on maps", "convlstm": "ConvLSTM on maps",
         "cnn_tuned": "CNN, tuned on pre-1984 data", "convlstm_tuned": "ConvLSTM, tuned on pre-1984 data",
         "ridge_ro": "Ridge"}


def selected_tuned_forecasts() -> pd.DataFrame:
    """Load only the pre-1984 rank-one E23 configurations, never the test-selected bound.

    Requires e23_selection.csv and every outer checkpoint for 1984-2026. Missing
    artifacts are errors so summary commands cannot silently omit the tuned models.
    This loader needs no PyTorch and does not fit or select any model.
    """
    selection_path = TABLES / "e23_selection.csv"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Restore E23 selection and outer checkpoints before regenerating tables: {selection_path}")
    selection = pd.read_csv(selection_path)
    if not {"kind", "config", "rank"}.issubset(selection.columns):
        raise ValueError(f"{selection_path}: expected kind, config and rank columns")
    parts = []
    for kind in ("cnn", "convlstm"):
        chosen = selection[(selection.kind == kind) & (selection["rank"] == 1)]
        if len(chosen) != 1:
            raise ValueError(f"{selection_path}: expected exactly one rank-one {kind} configuration")
        config = float(chosen.config.iloc[0])
        if not np.isfinite(config) or config < 0 or not config.is_integer():
            raise ValueError(f"{selection_path}: invalid configuration for {kind}: {config}")
        folder = TABLES / "e23" / "outer" / f"{kind}_{int(config)}"
        for year in range(1984, 2027):
            path = folder / f"year_{year}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing selected E23 forecast checkpoint; restore it before scoring: {path}")
            frame = pd.read_csv(path, parse_dates=["start"])
            if frame.empty or not frame.model.eq(kind).all() or not frame.origin_year.eq(year).all():
                raise ValueError(f"{path}: expected nonempty {kind} forecasts for origin year {year}")
            parts.append(frame[FORECAST_COLUMNS].assign(model=f"{kind}_tuned"))
    return pd.concat(parts, ignore_index=True)


def main_model_forecasts() -> pd.DataFrame:
    """Saved forecasts of all ten main models (E3, E7, E8, E13 and selected E23)."""
    def read(name):
        return pd.read_csv(TABLES / name, parse_dates=["start"])

    e7, e13, e8, e3, bb = (read("e7_forecasts.csv"), read("e13_forecasts.csv"), read("e8_forecasts.csv"),
                           read("e3_ro_forecasts.csv"), read("e3_ro_backbone_forecasts.csv"))
    return pd.concat([e13[e13.model == "ro_dmi"][FORECAST_COLUMNS],
                      e7[e7.label == "cs_lim|ro"].assign(model="recharge_oscillator")[FORECAST_COLUMNS],
                      bb[bb.model == "st"].assign(model="net_on_oscillator")[FORECAST_COLUMNS],
                      e3[e3.model == "st"].assign(model="net_ro")[FORECAST_COLUMNS],
                      e3[e3.model == "mt_auto"].assign(model="net_ro_mt")[FORECAST_COLUMNS],
                      e8[e8.model.isin(["cnn", "convlstm"])][FORECAST_COLUMNS],
                      selected_tuned_forecasts(),
                      e7[e7.label == "ridge|ro"].assign(model="ridge_ro")[FORECAST_COLUMNS]],
                     ignore_index=True)


def three_month_forecasts(d: pd.DataFrame) -> pd.DataFrame:
    """Average adjacent leads of the same model and start, retaining complete windows only."""
    if d.duplicated(["model", "start", "lead"]).any():
        raise ValueError("Three-month targets require unique model/start/lead forecasts")
    rows = []
    for (model, start), g in d.groupby(["model", "start"]):
        g = g.set_index("lead")
        for tau in g.index:
            window = [tau - 1, tau, tau + 1]
            if all(k in g.index for k in window):
                rows.append({"model": model, "start": start, "origin_year": g.origin_year.iloc[0],
                             "start_month": g.start_month.iloc[0], "lead": tau,
                             "forecast": g.loc[window, "forecast"].mean(), "observed": g.loc[window, "observed"].mean()})
    return pd.DataFrame(rows, columns=FORECAST_COLUMNS)


def seasonal_target() -> None:
    """Skill for a three-month-mean Niño-3.4 target, as used by most published deep-learning studies.

    For each start, the forecasts at leads tau-1, tau and tau+1 are averaged and verified against the
    mean of the observed anomalies at those leads, so no model is refitted. The centre lead tau is
    reported. Writes reports/tables/seasonal_target_skill.csv (ACC with 95% block-bootstrap
    intervals) and seasonal_target_vs_oscillator.csv."""
    seasonal = three_month_forecasts(main_model_forecasts())
    centre = (3, 6, 9, 11, 12, 18, 23)
    skill = pd.DataFrame([{"model": m, "lead": lead, **bootstrap_skill(seasonal, m, lead)}
                          for m in S1_NAMES for lead in centre if ((seasonal.model == m) & (seasonal.lead == lead)).any()])
    skill.round(4).to_csv(TABLES / "seasonal_target_skill.csv", index=False)
    diff = mark_significant(pd.DataFrame([
        {"model": m, "lead": lead, **bootstrap_difference(seasonal, m, "recharge_oscillator", lead)}
        for m in S1_NAMES if m != "recharge_oscillator" for lead in centre
        if ((seasonal.model == m) & (seasonal.lead == lead)).any()]))
    diff.round(4).to_csv(TABLES / "seasonal_target_vs_oscillator.csv", index=False)
    print(skill.pivot(index="model", columns="lead", values="acc").round(2).to_string())
    print(diff.round(3).to_string(index=False))


def table_s1() -> None:
    """Supplementary Table S1: ACC with 95% block-bootstrap intervals for the main models.

    Uses saved forecasts (E3, E7, E8, E13 and selected E23). Writes reports/tables/table_s1.csv."""
    d = main_model_forecasts()
    rows = []
    for model in S1_NAMES:
        for lead in LEADS:
            if ((d.model == model) & (d.lead == lead)).any():
                rows.append({"model": model, "lead": lead, **bootstrap_skill(d, model, lead)})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TABLES / "table_s1.csv", index=False)
    print(t.pivot(index="model", columns="lead", values="acc").round(2).to_string())


def paired_comparison(frame: pd.DataFrame, model: str, reference: str, lead: int, n_boot: int = 2000) -> list[dict]:
    """ACC and RMSE differences on identical starts, rejecting inconsistent observed targets."""
    pair = frame[frame.model.isin([model, reference]) & (frame.lead == lead)].copy()
    if pair.duplicated(["model", "start"]).any():
        raise ValueError(f"Duplicate forecasts in {model} versus {reference}, lead {lead}")
    pair = pair[np.isfinite(pair[["forecast", "observed"]]).all(axis=1)]
    a = pair[pair.model == model].set_index("start")
    b = pair[pair.model == reference].set_index("start")
    common = a.index.intersection(b.index).sort_values()
    if not len(common):
        raise ValueError(f"No common valid starts: {model} versus {reference}, lead {lead}")
    a, b = a.loc[common], b.loc[common]
    if not np.allclose(a.observed, b.observed, rtol=0, atol=1e-8) or not np.array_equal(a.origin_year, b.origin_year):
        raise ValueError(f"Paired observations/origin years disagree: {model} versus {reference}, lead {lead}")
    if a.origin_year.nunique() < 3:
        raise ValueError("At least three origin years are needed for the three-year block bootstrap")
    paired = pd.concat([a.reset_index(), b.reset_index()], ignore_index=True)
    common_info = {"model": model, "vs": reference, "lead": lead, "n": len(common),
                   "start_first": common.min(), "start_last": common.max(),
                   "origin_year_first": int(a.origin_year.min()), "origin_year_last": int(a.origin_year.max()),
                   "acc_model": anomaly_correlation(a.forecast, a.observed),
                   "acc_reference": anomaly_correlation(b.forecast, b.observed),
                   "rmse_model": float(np.sqrt(np.mean((a.forecast - a.observed) ** 2))),
                   "rmse_reference": float(np.sqrt(np.mean((b.forecast - b.observed) ** 2))),
                   "primary": lead in (18, 24)}
    rows = []
    for metric in ("acc", "rmse"):
        diff = bootstrap_difference(paired, model, reference, lead, metric=metric, n_boot=n_boot)
        if metric == "rmse":  # report error(model) - error(reference), lower is better
            diff = {"difference": -diff["difference"], "ci_low": -diff["ci_high"], "ci_high": -diff["ci_low"]}
        rows.append({**common_info, "metric": metric, **diff})
    return rows


def e24_extension() -> None:
    """E24: collect the separately trained 24-month index networks and paired E7 comparisons.

    Requires all 43 origin checkpoints and manifests for both E24 variants, plus E7.
    Add --with-gridded to require and compare E8 and training-selected E23 forecasts.
    Writes only e24_*.csv; never changes E3/E3b tables or manuscript values.
    """
    parts = []
    for variant, folder, labels in (
        ("e24_ro", "ro", {"st": "net_ro24", "mt_auto": "net_ro24_mt",
                             "mt_w0.25": "net_ro24_mt_w0.25", "mt_w0.5": "net_ro24_mt_w0.5", "mt_w1": "net_ro24_mt_w1"}),
        ("e24_backbone", "backbone", {"st": "net_backbone24"}),
    ):
        path = TABLES / "e24_2026-10-29" / folder
        manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))["identity"]
        if manifest["variant"] != variant or manifest["config"]["leads"] != list(range(1, 25)):
            raise ValueError(f"{path}: expected a separately labelled 1-24-month E24 run")
        if manifest["config"]["seeds"] != list(range(10)):
            raise ValueError(f"{path}: E24 requires all ten fixed seeds")
        for year in range(1984, 2027):
            checkpoint = path / f"year_{year}.csv"
            f = pd.read_csv(checkpoint, parse_dates=["start"])
            if f.empty or not f.origin_year.eq(year).all() or not set(labels).issubset(set(f.model)):
                raise ValueError(f"{checkpoint}: incomplete origin or model labels")
            parts.append(select_forecasts(f, "model", labels))
    e7 = pd.read_csv(TABLES / "e7_forecasts.csv", parse_dates=["start"])
    parts.append(select_forecasts(e7, "label", {"cs_lim|ro": "recharge_oscillator", "ridge|ro": "ridge_ro"}))
    references = ["recharge_oscillator", "ridge_ro"]
    if "--with-gridded" in sys.argv:
        e8 = pd.read_csv(TABLES / "e8_forecasts.csv", parse_dates=["start"])
        parts.extend([e8[e8.model.isin(["cnn", "convlstm"])][FORECAST_COLUMNS], selected_tuned_forecasts()])
        references += ["cnn", "convlstm", "cnn_tuned", "convlstm_tuned"]
    else:
        print("Gridded context not included; use --with-gridded after restoring E8/E23 artifacts.")
    forecasts = pd.concat(parts, ignore_index=True)
    if forecasts.duplicated(["model", "start", "lead"]).any():
        raise ValueError("E24 collection has duplicate model/start/lead forecasts")
    for model, f in forecasts.groupby("model"):
        if set(f.lead) != set(range(1, 25)):
            raise ValueError(f"{model}: missing forecast leads in E24 comparison")
    pairs = [("net_ro24_mt", "net_ro24")]
    pairs += [(m, r) for m in ("net_ro24", "net_ro24_mt", "net_backbone24") for r in references]
    pairs += [(m, "net_ro24") for m in sorted(set(forecasts.model)) if m.startswith("net_ro24_mt_w")]
    comparisons = mark_significant(pd.DataFrame([row for model, ref in pairs for lead in LEADS
                                                 for row in paired_comparison(forecasts, model, ref, lead)]))
    # Validate pairings before writing any result files. Marginal scores are labelled
    # separately; the comparison table above carries paired scores, counts and dates.
    forecasts.to_csv(TABLES / "e24_forecasts.csv", index=False)
    skill_table(forecasts).round(4).to_csv(TABLES / "e24_skill_by_lead.csv", index=False)
    skill_table(forecasts, by=("model", "start_month", "lead")).round(4).to_csv(
        TABLES / "e24_skill_by_start_month.csv", index=False)
    comparisons.round(4).to_csv(TABLES / "e24_comparisons.csv", index=False)
    intervals = pd.DataFrame([{"model": m, "lead": lead, **bootstrap_skill(forecasts, m, lead)}
                              for m in sorted(set(forecasts.model)) for lead in LEADS])
    intervals.round(4).to_csv(TABLES / "e24_acc_intervals.csv", index=False)
    seasonal = three_month_forecasts(forecasts)
    skill_table(seasonal).round(4).to_csv(TABLES / "e24_seasonal_skill.csv", index=False)
    print(comparisons[comparisons.primary].round(4).to_string(index=False))


COMMANDS = {
    "e24": e24_extension,
    "e6-seasons": e6_seasons,
    "e9-rules": e9_rules,
    "e10-p3": e10_p3_selection,
    "e11": e11_phase_scores,
    "e12": e12_combination,
    "e16": e16_events,
    "e22": e22_equivalence,
    "summary-figure": summary_figure,
    "cnn-figure": cnn_figure,
    "table-s1": table_s1,
    "seasonal-target": seasonal_target,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        names = "\n".join(f"  {name:<16} {function.__doc__.splitlines()[0]}" for name, function in COMMANDS.items())
        sys.exit(f"usage: python {Path(sys.argv[0]).name} COMMAND [options]\n\ncommands:\n{names}")
    command = sys.argv.pop(1)  # the remaining arguments are read by the command, as in the original scripts
    if "--check-inputs" not in sys.argv:
        ensure_output_directories()
    COMMANDS[command]()


if __name__ == "__main__":
    main()
