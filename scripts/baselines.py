"""Observed predictability and baseline forecasts of the monthly Niño-3.4 anomaly, rolling origin 1984-2026.

    python scripts/baselines.py COMMAND [options]

  lead-lag   descriptive lead-lag and partial correlations, ENSO events
  e1         ridge regression and LIMs for four input sets
  e2         SVR, random forest, gradient boosting and a scikit-learn network on SST + WWV
  e6         ERSST v5 and relative-index versions of the seasonal oscillator
  e7         linear models to 24 months, skill by start month
  e9         E9 linear part: SST record from 1880 vs from 1958 (development years; --confirm for 2006-2026)
  e20        E20 linear part: ridge regression fitted separately for each start month

Every origin year refits climatologies, scaling and models on earlier years only (see
src/enso_climate/forecast.py). Inputs: data/processed/ (scripts/prepare_data.py build, and
long-indices for e9). Project environment. Writes reports/tables/ and reports/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enso_climate.paths import ensure_output_directories  # noqa: E402

from enso_climate.forecast import (  # noqa: E402
    INPUT_SETS, TRAIN_START, Climatology, FeatureSpec, MonthlyRidge, RidgeModel, RollingConfig, SklearnModel, bootstrap_difference,
    bootstrap_skill, load_inputs, load_monthly, load_quarterly, mark_significant, rolling_forecasts, skill_table,
)
from enso_climate.plotting import (  # noqa: E402
    AMBER, AXIS, BLUE, GREEN, GRID, INK, INK2, LIGHT_BLUE, MONTHS, MUTED, ORANGE, PINK, SERIES, plot_acc_rmse, style,
    suptitle,
)

TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
LEADS = tuple(range(1, 25))


# ----------------------------------------------------------------------------
# lead-lag: descriptive analysis of the dataset (no model is fitted)
# ----------------------------------------------------------------------------

TARGET = "nino34_ssta"
WARM_SIDE, COOL_SIDE = ORANGE, BLUE

GROUPS = {
    "SST (ERSST v6)": {"wtio_ssta": "WTIO", "etio_ssta": "ETIO", "dmi": "DMI"},
    "Zonal wind (CCMP)": {"u_wpac_anom": "W Pacific", "u_cpac_anom": "C Pacific", "u_epac_anom": "E Pacific", "u_eqio_anom": "Eq. Indian"},
    "Precipitation (ERA5)": {"pr_nino4_anom": "Niño-4", "pr_nino34_anom": "Niño-3.4", "pr_wtio_anom": "WTIO", "pr_etio_anom": "ETIO"},
}
OHC = {"ohc700_eqpac": "OHC 0-700 m, eq. Pacific", "ohc700_nino34": "OHC 0-700 m, Niño-3.4", "ohc2000_nino34": "OHC 0-2000 m, Niño-3.4"}


def lag1(x: pd.Series) -> float:
    return x.autocorr(1)


def critical_r(x: pd.Series, y: pd.Series) -> float:
    n = len(x)
    a, b = lag1(x), lag1(y)
    n_eff = max(n * (1 - a * b) / (1 + a * b), 4)
    return 1.96 / np.sqrt(n_eff - 2)


def partial_r(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    rxy, rxz, ryz = x.corr(y), x.corr(z), y.corr(z)
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))


def lead_lag(frame: pd.DataFrame, predictors, target: str, leads) -> pd.DataFrame:
    rows = []
    for column in predictors:
        for lead in leads:
            pair = pd.DataFrame({"x": frame[column], "y": frame[target].shift(-lead), "z": frame[target]}).dropna()
            rows.append({
                "predictor": column,
                "lead": lead,
                "n": len(pair),
                "start": str(pair.index.min())[:10],
                "end": str(pair.index.max())[:10],
                "r": pair.x.corr(pair.y),
                "partial_r_given_target_now": np.nan if column == target or lead == 0 else partial_r(pair.x, pair.y, pair.z),
                "r_95pct_threshold": critical_r(pair.x, pair.y),
            })
    return pd.DataFrame(rows)


def events(monthly: pd.DataFrame) -> pd.DataFrame:
    phase = monthly["enso_phase_relative"].fillna("neutral")
    run = (phase != phase.shift()).cumsum()
    rows = []
    for _, block in monthly.groupby(run):
        label = phase.loc[block.index[0]]
        if label == "neutral":
            continue
        peak = block["roni"].abs().idxmax()
        rows.append({
            "phase": label,
            "start": block.index[0].strftime("%Y-%m"),
            "end": block.index[-1].strftime("%Y-%m"),
            "months": len(block),
            "peak_month": peak.strftime("%Y-%m"),
            "peak_roni_C": round(block.loc[peak, "roni"], 2),
            "peak_oni_C": round(block.loc[peak, "oni"], 2),
            "ongoing_at_end_of_data": block.index[-1] >= monthly["roni"].last_valid_index(),
        })
    return pd.DataFrame(rows)


def plot_history(monthly, path):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    roni = monthly["roni"].dropna()
    ax.fill_between(roni.index, 0, roni.where(roni > 0, 0), color=WARM_SIDE, linewidth=0, label="Warm (El Niño side)")
    ax.fill_between(roni.index, 0, roni.where(roni < 0, 0), color=COOL_SIDE, linewidth=0, label="Cool (La Niña side)")
    for level in (0.5, -0.5):
        ax.axhline(level, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.axhline(0, color=AXIS, linewidth=0.8)
    style(ax, "Relative ONI, 1950–2026 (ERSST v6, 1991–2020 base; dashed = ±0.5 °C)")
    ax.set_ylabel("°C", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lead_lag(table, path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey="row")
    for col, (group, names) in enumerate(GROUPS.items()):
        for row, metric in enumerate(("r", "partial_r_given_target_now")):
            ax = axes[row, col]
            for colour, (column, label) in zip(SERIES, names.items()):
                part = table[table.predictor == column]
                if part[metric].isna().all():
                    continue
                ax.plot(part.lead, part[metric], color=colour, linewidth=2, marker="o", markersize=3.5, label=label)
            threshold = table[table.predictor.isin(names)].r_95pct_threshold.max()
            ax.axhspan(-threshold, threshold, color=GRID, alpha=0.6, linewidth=0)
            ax.axhline(0, color=AXIS, linewidth=0.8)
            style(ax, group if row == 0 else f"{group}: beyond Niño-3.4 now")
            if row == 0:
                ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
            else:
                ax.set_xlabel("Lead of Niño-3.4 target (months)", color=INK2, fontsize=9)
        axes[0, 0].set_ylabel("Correlation r", color=INK2, fontsize=9)
        axes[1, 0].set_ylabel("Partial r given Niño-3.4 now", color=INK2, fontsize=9)
    suptitle(fig, "Predictor at month t vs Niño-3.4 anomaly at t + lead (shaded: not significant at 95%, effective n)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ohc(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, metric, title in zip(axes, ("r", "partial_r_given_target_now"), ("Correlation", "Beyond Niño-3.4 now (partial r)")):
        for colour, (column, label) in zip(SERIES, OHC.items()):
            part = table[table.predictor == column]
            ax.plot(part.lead, part[metric], color=colour, linewidth=2, marker="o", markersize=4, label=f"{label} (n≈{part.n.max()})")
        threshold = table[table.predictor != "ohc2000_nino34"].r_95pct_threshold.max()
        ax.axhspan(-threshold, threshold, color=GRID, alpha=0.6, linewidth=0)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        style(ax, title)
        ax.set_xticks(range(0, 5))
        ax.set_xlabel("Lead of Niño-3.4 target (quarters)", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    suptitle(fig, "Ocean heat content (quarter t) vs Niño-3.4 anomaly (quarter t + lead); shaded: n.s. for 0-700 m (0-2000 m needs |r| > 0.43)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def describe_lead_lag() -> None:
    """Descriptive lead-lag analysis of the dataset; no model is fitted.

    Reads data/processed/enso_monthly.csv and enso_quarterly.csv. Writes:
      reports/tables/enso_events.csv          El Niño / La Niña events from relative ONI
      reports/tables/lead_lag_monthly.csv     r and partial r (given Niño-3.4 at t) by lead
      reports/tables/lead_lag_quarterly.csv   same for ocean heat content, in quarters
      reports/figures/*.png
    Correlation significance uses an effective sample size (Bretherton et al. 1999)
    because monthly anomalies are strongly autocorrelated."""
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    monthly = load_monthly()
    quarterly = load_quarterly()

    event_table = events(monthly)
    event_table.to_csv(TABLES / "enso_events.csv", index=False)

    predictors = [c for names in GROUPS.values() for c in names]
    monthly_table = lead_lag(monthly, predictors, TARGET, range(0, 13))
    monthly_table.round(4).to_csv(TABLES / "lead_lag_monthly.csv", index=False)

    quarterly_table = lead_lag(quarterly, list(OHC), f"{TARGET}_qmean", range(0, 5))
    quarterly_table.round(4).to_csv(TABLES / "lead_lag_quarterly.csv", index=False)

    plot_history(monthly, FIGURES / "enso_relative_oni_1950_2026.png")
    plot_lead_lag(monthly_table, FIGURES / "lead_lag_monthly.png")
    plot_ohc(quarterly_table, FIGURES / "lead_lag_ohc_quarterly.png")

    print(event_table.groupby("phase").agg(events=("months", "size"), mean_months=("months", "mean"), max_peak=("peak_roni_C", lambda s: s.abs().max())))
    pick = monthly_table[monthly_table.lead.isin([0, 3, 6, 9, 12])]
    print(pick.pivot(index="predictor", columns="lead", values="r").round(2))
    print(pick.pivot(index="predictor", columns="lead", values="partial_r_given_target_now").round(2))
    print(pick.groupby("predictor")[["n", "r_95pct_threshold"]].max().round(2))
    print(quarterly_table.round(2).to_string(index=False))


# ----------------------------------------------------------------------------
# e1: baselines for four input sets
# ----------------------------------------------------------------------------


def plot_seasonal(forecasts: pd.DataFrame, panels: dict[str, str], path: Path) -> None:
    """panels: label -> panel title."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), sharey=True)
    for ax, (label, title) in zip(axes, panels.items()):
        part = forecasts[forecasts.label == label]
        table = skill_table(part, by=("start_month", "lead")).pivot(index="start_month", columns="lead", values="acc")
        image = ax.imshow(table.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(12), table.columns)
        ax.set_yticks(range(12), MONTHS)
        ax.set_xlabel("Lead (months)", color=INK2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Start month", color=INK2, fontsize=9)
    bar = fig.colorbar(image, ax=axes, fraction=0.025)
    bar.set_label("Anomaly correlation (≤ 0 white)", color=INK2, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def e1_baselines() -> None:
    """E1: baseline forecasts of the monthly Niño-3.4 anomaly, rolling origin 1984-2026.

    Writes reports/tables/e1_*.csv and reports/figures/e1_*.png."""
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frames = load_inputs()
    point = [Climatology()]
    runs = []
    for name, (frame_key, lags, lim_state) in INPUT_SETS.items():
        config = RollingConfig(first_test_year=1984, last_test_year=2026, train_start=TRAIN_START,
                               lim_state=lim_state or ("nino34",))
        models = (point if name == "sst" else []) + [RidgeModel()]
        out = rolling_forecasts(frames[frame_key], FeatureSpec(lags=lags), config, models)
        if lim_state is None:
            out = out[~out.model.isin(["lim", "cs_lim"])]
        runs.append(out.assign(inputs=name))
        print(f"{name}: done", flush=True)
    forecasts = pd.concat(runs, ignore_index=True)
    forecasts["label"] = forecasts.model + "|" + forecasts.inputs
    forecasts.to_csv(TABLES / "e1_forecasts.csv", index=False)

    skill = skill_table(forecasts, by=("label", "model", "inputs", "lead"))
    skill.round(4).to_csv(TABLES / "e1_skill_by_lead.csv", index=False)
    skill_table(forecasts, by=("label", "start_month", "lead")).round(4).to_csv(TABLES / "e1_skill_by_start_month.csv", index=False)

    reference = "cs_lim|ro"  # the seasonal recharge oscillator
    rows = []
    for label in sorted(set(forecasts.label) - {reference, "climatology|sst"}):
        sub = forecasts[forecasts.label.isin([label, reference])].assign(model=lambda d: d.label)
        for lead in range(1, 13):
            rows.append({"label": label, "lead": lead, **bootstrap_difference(sub, label, reference, lead)})
    diff = mark_significant(pd.DataFrame(rows))
    diff.round(4).to_csv(TABLES / "e1_vs_oscillator.csv", index=False)

    plot_acc_rmse(skill, {
        "ridge|sst+wwv": ("Ridge (SST + monthly WWV)", BLUE, "-"),
        "cs_lim|sst+wwv": ("Cyclostationary LIM (SST + WWV)", ORANGE, "-"),
        "cs_lim|ro": ("Linear recharge oscillator (Niño-3.4 + WWV)", GREEN, "-"),
        "ridge|sst+ohcq": ("Ridge (SST + quarterly heat content)", BLUE, (0, (4, 3))),
    }, FIGURES / "e1_skill_by_lead.png",
        "Monthly Niño-3.4 anomaly forecasts, rolling origin 1984–2026 (training-only preprocessing)")
    plot_seasonal(forecasts, {"cs_lim|ro": "Seasonal recharge oscillator", "cs_lim|sst+wwv": "Cyclostationary LIM (SST + WWV)",
                              "ridge|sst+wwv": "Ridge (SST + WWV)"}, FIGURES / "e1_skill_by_start_month.png")

    print(skill.pivot_table(index="label", columns="lead", values="acc").round(2).to_string())
    print(skill.pivot_table(index="label", columns="lead", values="rmse").round(2).to_string())
    print(diff[diff.lead.isin([3, 6, 9, 12])].round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# e2: machine-learning models
# ----------------------------------------------------------------------------


def svr(random_state=0, C=1.0, gamma="scale", epsilon=0.1):
    return make_pipeline(StandardScaler(), SVR(C=C, gamma=gamma, epsilon=epsilon))


def forest(random_state=0, min_samples_leaf=5):
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=min_samples_leaf, max_features=0.5,
                                 n_jobs=1, random_state=random_state)


def boosting(random_state=0, learning_rate=0.05, max_depth=3):
    return HistGradientBoostingRegressor(learning_rate=learning_rate, max_depth=max_depth, max_iter=200,
                                         l2_regularization=1.0, random_state=random_state)


def mlp(random_state=0, alpha=1.0):
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,), alpha=alpha, max_iter=2000,
                                                       learning_rate_init=1e-3, random_state=random_state))


E2_MODELS = [
    RidgeModel(),
    SklearnModel("svr", svr, grid=[{"C": c, "gamma": g} for c in (0.3, 1.0, 3.0) for g in ("scale", 0.01)]),
    SklearnModel("random_forest", forest, grid=[{"min_samples_leaf": k} for k in (5, 20)]),
    SklearnModel("gradient_boosting", boosting, grid=[{"learning_rate": r, "max_depth": d} for r in (0.03, 0.1) for d in (2, 3)]),
    SklearnModel("mlp_single_task", mlp, grid=[{"alpha": a} for a in (0.1, 1.0, 10.0)], seeds=(0, 1, 2, 3, 4)),
]
E2_COLOURS = {"ridge": BLUE, "svr": ORANGE, "random_forest": GREEN, "gradient_boosting": AMBER,
           "mlp_single_task": PINK}
E2_LABELS = {"ridge": "Ridge", "svr": "SVR", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
          "mlp_single_task": "Single-task network (MLP)"}


def e2_machine_learning() -> None:
    """E2: machine-learning forecasts of the monthly Niño-3.4 anomaly, rolling origin 1984-2026.

    Same protocol and inputs (SST indices + monthly ORAS5 warm water volume) as the best
    E1 baseline. Hyperparameters are chosen by chronological CV inside each training
    window; the neural network is averaged over five seeds fixed in advance. Writes
    reports/tables/e2_*.csv and reports/figures/e2_skill_by_lead.png."""
    frames = load_inputs()
    config = RollingConfig(first_test_year=1984, last_test_year=2026, train_start=TRAIN_START)
    spec = FeatureSpec(lags=INPUT_SETS["sst+wwv"][1])
    forecasts = rolling_forecasts(frames["wwv"], spec, config, E2_MODELS)
    forecasts = forecasts[~forecasts.model.isin(["lim", "cs_lim"])].assign(inputs="sst+wwv")
    forecasts.to_csv(TABLES / "e2_forecasts.csv", index=False)

    skill = skill_table(forecasts, by=("model", "lead"))
    skill.round(4).to_csv(TABLES / "e2_skill_by_lead.csv", index=False)
    rows = []
    for model in E2_COLOURS:
        if model == "ridge":
            continue
        for lead in config.leads:
            rows.append({"model": model, "vs": "ridge", "lead": lead, **bootstrap_difference(forecasts, model, "ridge", lead)})
    pd.DataFrame(rows).round(4).to_csv(TABLES / "e2_vs_ridge.csv", index=False)

    plot_acc_rmse(skill, {model: (E2_LABELS[model], colour, "-") for model, colour in E2_COLOURS.items()},
                  FIGURES / "e2_skill_by_lead.png",
                  "Machine-learning vs linear forecasts (SST + monthly warm water volume), rolling origin 1984–2026",
                  key="model")

    print(skill.pivot(index="model", columns="lead", values="acc").round(2).to_string())
    print(skill.pivot(index="model", columns="lead", values="rmse").round(2).to_string())
    d = pd.DataFrame(rows)
    d["sig"] = (d.ci_low > 0) | (d.ci_high < 0)
    print(d[d.lead.isin([3, 6, 9, 12])].round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# e6: robustness to the SST product and the index definition
# ----------------------------------------------------------------------------


def e6_robustness() -> None:
    """E6: robustness of the headline comparison to the SST product and the index definition.

    The seasonal recharge oscillator and ridge regression on the same lagged Niño-3.4 and WWV
    are re-run with (a) ERSST v5 Niño-3.4 instead of v6, and (b) relative Niño-3.4 (minus the
    tropical-mean SST, the RONI idea). Same protocol as E7, leads 1-24. Writes
    reports/tables/e6_robustness.csv and e6_robustness_vs_ridge.csv."""
    base = load_inputs()["wwv"]
    monthly = load_monthly()
    versions = {
        "v6": base[["nino34", "d20_wwv"]],
        "v5": base[["d20_wwv"]].join(monthly["nino34_sst_v5"].rename("nino34")),
        "relative_v6": base[["d20_wwv"]].assign(nino34=base["nino34"] - base["tropics"]),
    }
    runs = []
    for name, data in versions.items():
        config = RollingConfig(first_test_year=1984, last_test_year=2026, leads=LEADS, train_start=TRAIN_START,
                               lim_state=("nino34", "d20_wwv"))
        out = rolling_forecasts(data[["nino34", "d20_wwv"]], FeatureSpec(lags=INPUT_SETS["ro"][1]), config, [RidgeModel()])
        out = out[out.model.isin(["cs_lim", "ridge"])]
        runs.append(out.assign(model=out.model + "|" + name))
    forecasts = pd.concat(runs, ignore_index=True)
    skill = skill_table(forecasts, by=("model", "lead"))
    rows = [{"version": v, "lead": lead, **bootstrap_difference(forecasts, f"cs_lim|{v}", f"ridge|{v}", lead)}
            for v in versions for lead in (6, 12, 18, 24)]
    diff = mark_significant(pd.DataFrame(rows))
    skill.round(4).to_csv(TABLES / "e6_robustness.csv", index=False)
    diff.round(4).to_csv(TABLES / "e6_robustness_vs_ridge.csv", index=False)
    print(skill.pivot(index="model", columns="lead", values="acc")[[3, 6, 9, 12, 18, 24]].round(2).to_string())
    print(diff.round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# e7: linear models to 24 months
# ----------------------------------------------------------------------------

E7_LINES = {
    "cs_lim|ro": ("Seasonal recharge oscillator (Niño-3.4 + WWV)", ORANGE),
    "cs_lim|sst+wwv": ("Cyclostationary LIM (SST + WWV)", BLUE),
    "lim|ro": ("Stationary LIM (Niño-3.4 + WWV)", GREEN),
    "cs_lim|sst": ("Cyclostationary LIM (SST only)", LIGHT_BLUE),
    "ridge|ro": ("Ridge (Niño-3.4 + WWV)", PINK),
}


def e7_long_leads() -> None:
    """E7: how far does observation-only Niño-3.4 skill extend? Linear baselines to 24 months.

    Same rolling-origin protocol and inputs as E1 (test years 1984-2026), with leads 1-24.
    Each model's anomaly correlation carries a 3-year block-bootstrap interval, so the lead
    at which skill can no longer be told apart from zero is reported, overall and by start
    month. The oscillator is compared with each other model in e7_oscillator_vs_linear.csv.
    Writes reports/tables/e7_*.csv and reports/figures/e7_*.png."""
    frames = load_inputs()
    runs = []
    for name in ("sst", "sst+wwv", "ro"):
        frame_key, lags, lim_state = INPUT_SETS[name]
        config = RollingConfig(first_test_year=1984, last_test_year=2026, leads=LEADS, train_start=TRAIN_START,
                               lim_state=lim_state)
        models = [] if name == "sst" else [RidgeModel()]  # the LIMs are fitted for every input set
        runs.append(rolling_forecasts(frames[frame_key], FeatureSpec(lags=lags), config, models).assign(inputs=name))
        print(f"{name}: done", flush=True)
    forecasts = pd.concat(runs, ignore_index=True)
    forecasts["label"] = forecasts.model + "|" + forecasts.inputs
    forecasts = forecasts[forecasts.label.isin(E7_LINES)]
    forecasts.to_csv(TABLES / "e7_forecasts.csv", index=False)
    by_label = forecasts.assign(model=forecasts.label)

    skill = skill_table(by_label, by=("model", "lead")).rename(columns={"model": "label"})
    ci = pd.DataFrame([{"label": label, "lead": lead, **bootstrap_skill(by_label, label, lead)}
                       for label in E7_LINES for lead in LEADS])
    skill = skill.merge(ci[["label", "lead", "ci_low", "ci_high"]], on=["label", "lead"])
    skill.round(4).to_csv(TABLES / "e7_skill_by_lead.csv", index=False)
    seasonal = skill_table(by_label, by=("model", "start_month", "lead")).rename(columns={"model": "label"})
    seasonal.round(4).to_csv(TABLES / "e7_skill_by_start_month.csv", index=False)
    rows = [{"label": "cs_lim|ro", "vs": label, "lead": lead, **bootstrap_difference(by_label, "cs_lim|ro", label, lead)}
            for label in E7_LINES if label != "cs_lim|ro" for lead in LEADS]
    diff = mark_significant(pd.DataFrame(rows))
    diff.round(4).to_csv(TABLES / "e7_oscillator_vs_linear.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for label, (name, colour) in E7_LINES.items():
        part = skill[skill.label == label]
        ax.plot(part.lead, part.acc, color=colour, linewidth=2, marker="o", markersize=3, label=name)
        if label == "cs_lim|ro":
            ax.fill_between(part.lead, part.ci_low, part.ci_high, color=colour, alpha=0.15, linewidth=0)
    ax.axhline(0.5, color=AXIS, linewidth=0.8)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    style(ax, "Anomaly correlation (band: 95% block-bootstrap interval, seasonal recharge oscillator)")
    ax.set_xlabel("Lead (months)", color=INK2, fontsize=9)
    ax.set_xticks(range(1, 25))
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    suptitle(fig, "How far observation-only Niño-3.4 skill extends: linear models to 24 months, 1984–2026")
    fig.tight_layout()
    fig.savefig(FIGURES / "e7_skill_by_lead.png", dpi=150)
    plt.close(fig)

    table = seasonal[seasonal.label == "cs_lim|ro"].pivot(index="start_month", columns="lead", values="acc")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    image = ax.imshow(table.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(12), MONTHS)
    ax.set_xlabel("Lead (months)", color=INK2, fontsize=9)
    ax.set_ylabel("Start month", color=INK2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.025)
    bar.set_label("Anomaly correlation (≤ 0 white)", color=INK2, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title("Seasonal recharge oscillator: skill by start month and lead, 1984–2026", loc="left", fontsize=10, color=INK)
    fig.savefig(FIGURES / "e7_skill_by_start_month.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(skill.pivot(index="label", columns="lead", values="acc").round(2).to_string())
    print(skill[skill.label == "cs_lim|ro"][["lead", "acc", "ci_low", "ci_high"]].round(2).to_string(index=False))
    print(table[[6, 9, 12, 15, 18, 21, 24]].round(2).to_string())


# ----------------------------------------------------------------------------
# e20: ridge regression fitted separately for each start month
# ----------------------------------------------------------------------------


def e20_seasonal_ridge() -> None:
    """E20 (linear part): ridge regression fitted separately for each start month, leads 1-24.

    Same inputs (Niño-3.4 and WWV lags) and protocol as E7 "ro". Compared with pooled ridge and
    the seasonal recharge oscillator from E7. Writes reports/tables/e20_seasonal_ridge_forecasts.csv
    and e20_seasonal_ridge_vs_linear.csv."""
    frame_key, lags, _ = INPUT_SETS["ro"]
    config = RollingConfig(first_test_year=1984, last_test_year=2026, leads=LEADS, train_start=TRAIN_START,
                           lim_state=("nino34",))
    out = rolling_forecasts(load_inputs()[frame_key], FeatureSpec(lags=lags), config, [MonthlyRidge()])
    out = out[out.model == "seasonal_ridge"]
    out.to_csv(TABLES / "e20_seasonal_ridge_forecasts.csv", index=False)
    e7 = pd.read_csv(TABLES / "e7_forecasts.csv", parse_dates=["start"])
    refs = e7[e7.label.isin(["cs_lim|ro", "ridge|ro"])].assign(model=lambda d: d.label)[out.columns]
    both = pd.concat([out, refs], ignore_index=True)
    rows = [{"model": "seasonal_ridge", "vs": ref, "lead": lead, **bootstrap_difference(both, "seasonal_ridge", ref, lead)}
            for ref in ("ridge|ro", "cs_lim|ro") for lead in LEADS]
    diff = mark_significant(pd.DataFrame(rows))
    diff.round(4).to_csv(TABLES / "e20_seasonal_ridge_vs_linear.csv", index=False)
    print(skill_table(both, by=("model", "lead")).pivot(index="model", columns="lead", values="acc")[[3, 6, 9, 12, 18, 24]].round(2))
    print(diff[diff.lead.isin([3, 6, 9, 12, 18, 24])].round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# e9: the SST-only record from 1880 (linear part)
# ----------------------------------------------------------------------------

E9_SST = ("nino34", "wtio", "etio", "tropics")
E9_LEADS = tuple(range(1, 13))


def e9_lags(n_nino):
    return {"nino34": n_nino, "wtio": 3, "etio": 3, "tropics": 3}


def e9_long_record() -> None:
    """E9 (linear part): does the SST-only record from 1880 help?

    All models use training-only climatology plus training-only linear detrending. Runs the
    development test years (1984-2005) by default; --confirm runs 2006-2026 once the E9
    configuration is frozen. Writes reports/tables/e9_linear_<period>_*.csv."""
    period = "confirm" if "--confirm" in sys.argv else "dev"
    first, last = (2006, 2026) if period == "confirm" else (1984, 2005)
    long = load_inputs()["long"]
    runs = []
    for start, tag in (("1880-01-01", "long"), ("1958-01-01", "short")):
        for n in (12, 24):
            config = RollingConfig(first_test_year=first, last_test_year=last, leads=E9_LEADS, train_start=start,
                                   lim_state=E9_SST, detrend=True)
            models = [RidgeModel()]
            out = rolling_forecasts(long[list(E9_SST)], FeatureSpec(lags=e9_lags(n)), config, models)
            if n == 24:
                out = out[out.model == "ridge"]
            runs.append(out.assign(label=lambda d: d.model + f"|sst{n}|{tag}"))
            print(tag, n, "done", flush=True)
    ro = RollingConfig(first_test_year=first, last_test_year=last, leads=E9_LEADS, train_start="1958-01-01",
                       lim_state=("nino34", "d20_wwv"), detrend=True)
    out = rolling_forecasts(long[["nino34", "d20_wwv"]], FeatureSpec(lags={"nino34": 1}), ro, [])
    runs.append(out[out.model == "cs_lim"].assign(label="recharge_oscillator"))
    forecasts = pd.concat(runs, ignore_index=True)
    forecasts.loc[forecasts.label.str.startswith("cs_lim|sst12"), "label"] = forecasts.label.str.replace("sst12", "sst")
    forecasts = forecasts[~forecasts.label.str.contains(r"^(lim|cs_lim)\|sst24")]
    forecasts["model"] = forecasts["label"]
    forecasts.to_csv(TABLES / f"e9_linear_{period}_forecasts.csv", index=False)
    skill = skill_table(forecasts, by=("model", "lead"))
    skill.round(4).to_csv(TABLES / f"e9_linear_{period}_skill_by_lead.csv", index=False)
    pairs = [("ridge|sst12|long", "ridge|sst12|short"), ("ridge|sst24|long", "ridge|sst24|short"),
             ("cs_lim|sst|long", "cs_lim|sst|short")]
    rows = [{"model": a, "vs": b, "lead": lead, **bootstrap_difference(forecasts, a, b, lead)}
            for a, b in pairs for lead in (3, 6, 9, 12)]
    diff = mark_significant(pd.DataFrame(rows))
    diff.round(4).to_csv(TABLES / f"e9_linear_{period}_long_vs_short.csv", index=False)
    print(skill.pivot(index="model", columns="lead", values="acc")[[1, 3, 6, 9, 12]].round(2).to_string())
    print(diff.round(3).to_string(index=False))


COMMANDS = {
    "lead-lag": describe_lead_lag,
    "e1": e1_baselines,
    "e2": e2_machine_learning,
    "e6": e6_robustness,
    "e7": e7_long_leads,
    "e9": e9_long_record,
    "e20": e20_seasonal_ridge,
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
