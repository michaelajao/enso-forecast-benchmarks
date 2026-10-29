"""Experiments on the seasonal recharge oscillator (cyclostationary LIM on Niño-3.4 and WWV), test years 1984-2026.

    python scripts/oscillator.py COMMAND [options]

  e5            autumn IOD -> ENSO at 13 months, one forecast per year
  e13           three-variable oscillators: + DMI, + western WWV
  e14           + Atlantic indices ATL3 and NTA (reads the raw ERSST v6 file)
  e15           probabilistic skill: CRPS, coverage, spread-error
  e17           XRO-style quadratic terms
  e18           exact split of the forecast into Niño-3.4 and WWV parts
  e19-ablation  skill lost without WWV, oscillator and network (reads saved E3, E7, E10 forecasts)
  e21-pmel      the oscillator and ridge with PMEL in-situ warm water volume, 2000-2026
  operators     the fitted one-month operators by calendar month (descriptive, 1958-2025)

Every fit uses training years only (see src/enso_climate/forecast.py). Project environment
(e14 needs netCDF4). Writes reports/tables/e*_*.csv, ro_operator.csv and reports/figures/ro_operator.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enso_climate.paths import ensure_output_directories  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))

from enso_climate.data import box_mean, load_ersst  # noqa: E402
from enso_climate.forecast import (  # noqa: E402
    FORECAST_COLUMNS, TRAIN_START, FeatureSpec, RidgeModel, RollingConfig, anomalies_from_training,
    anomaly_correlation, block_resample, bootstrap_difference, crps_gaussian, cs_lim_forecasts,
    fit_cyclostationary_lim, load_inputs, mark_significant, rolling_forecasts, select_forecasts, skill_table,
)
from enso_climate.plotting import AXIS, BLUE, GREEN, INK2, MONTHS, ORANGE, PINK, style, suptitle  # noqa: E402

TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
LEADS = tuple(range(1, 25))


# ----------------------------------------------------------------------------
# e5: autumn IOD -> ENSO at 13 months
# ----------------------------------------------------------------------------


def yearly(anom: pd.DataFrame) -> pd.DataFrame:
    """Per start year y: SON predictors and the Nov(y+1)-Jan(y+2) Niño-3.4 target."""
    son = anom[anom.index.month.isin([9, 10, 11])]
    pred = son.groupby(son.index.year).mean()
    pred["dmi"] = pred["wtio"] - pred["etio"]
    ndj = anom["nino34"].rolling(3).mean()  # value at Jan = mean of Nov, Dec, Jan
    target = ndj[ndj.index.month == 1]
    target.index = target.index.year - 2  # Jan of y+2 belongs to start year y
    return pred[["nino34", "d20_wwv", "dmi"]].join(target.rename("target"))


def fit_predict(train: pd.DataFrame, test: pd.Series, columns) -> tuple[float, np.ndarray]:
    X = np.column_stack([np.ones(len(train)), train[list(columns)].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, train["target"].to_numpy(), rcond=None)
    return float(coef[0] + test[list(columns)].to_numpy() @ coef[1:]), coef


def e5_autumn_iod() -> None:
    """E5: chronological test of the autumn IOD -> ENSO link at 13-15 months.

    One forecast per year, starting November of year y with SON means; the target is the
    November-January Niño-3.4 anomaly of years y+1/y+2. Anomalies use training years only.
    Writes reports/tables/e5_*.csv."""
    raw = load_inputs()["wwv"].loc[TRAIN_START:]
    rows = []
    for y in range(1983, 2025):
        cutoff = pd.Timestamp(f"{y}-12-01")  # forecast made at the start of December, after SON
        anom = anomalies_from_training(raw, cutoff)
        table = yearly(anom)
        train = table.loc[: y - 2].dropna()  # targets end Jan(y) at the latest, before the cutoff
        if y not in table.index or np.isnan(table.loc[y, ["nino34", "d20_wwv", "dmi"]]).any():
            continue
        test = table.loc[y]
        ro, _ = fit_predict(train, test, ("nino34", "d20_wwv"))
        iod, coef = fit_predict(train, test, ("nino34", "d20_wwv", "dmi"))
        rows.append({"start_year": y, "observed": test["target"], "ro": ro, "ro_iod": iod, "dmi_coef": coef[-1],
                     "n_train": len(train)})
    out = pd.DataFrame(rows).dropna(subset=["observed"])
    out.to_csv(TABLES / "e5_forecasts.csv", index=False)
    rng = np.random.default_rng(0)
    years = out.start_year.to_numpy()

    def acc(frame, col):
        return anomaly_correlation(frame[col], frame.observed)

    draws = []
    for _ in range(4000):
        sample = out.iloc[block_resample(len(years), 3, rng)]
        draws.append(acc(sample, "ro_iod") - acc(sample, "ro"))
    low, high = np.percentile(draws, [2.5, 97.5])
    rmse = {c: float(np.sqrt(np.mean((out[c] - out.observed) ** 2))) for c in ("ro", "ro_iod")}
    summary = pd.DataFrame([{
        "n_test_years": len(out), "acc_ro": acc(out, "ro"), "acc_ro_iod": acc(out, "ro_iod"),
        "difference": acc(out, "ro_iod") - acc(out, "ro"), "ci_low": low, "ci_high": high,
        "rmse_ro": rmse["ro"], "rmse_ro_iod": rmse["ro_iod"],
        "share_negative_dmi_coef": float((out.dmi_coef < 0).mean()), "median_dmi_coef": float(out.dmi_coef.median()),
    }])
    summary.round(4).to_csv(TABLES / "e5_summary.csv", index=False)
    print(summary.round(3).T.to_string())


# ----------------------------------------------------------------------------
# e13 and e14: three- and four-variable oscillators
# ----------------------------------------------------------------------------

E13_STATES = {"ro": ("nino34", "d20_wwv"), "ro_dmi": ("nino34", "d20_wwv", "dmi"),
          "ro_west": ("nino34", "d20_wwv", "d20_wwv_west")}


def e13_three_variable() -> None:
    """E13: three-variable seasonal recharge oscillators against the two-variable one.

    Writes reports/tables/e13_*.csv."""
    data = load_inputs()["wwv"]
    data["dmi"] = data["wtio"] - data["etio"]
    forecasts = pd.concat([cs_lim_forecasts(data, state, name, LEADS, TRAIN_START) for name, state in E13_STATES.items()],
                          ignore_index=True)
    forecasts.to_csv(TABLES / "e13_forecasts.csv", index=False)
    skill = skill_table(forecasts, by=("model", "lead"))
    skill.round(4).to_csv(TABLES / "e13_skill_by_lead.csv", index=False)
    rows = [{"model": m, "lead": lead, **bootstrap_difference(forecasts, m, "ro", lead)}
            for m in ("ro_dmi", "ro_west") for lead in (3, 6, 9, 12, 15, 18, 24)]
    diff = mark_significant(pd.DataFrame(rows))
    diff.round(4).to_csv(TABLES / "e13_vs_ro.csv", index=False)
    print(skill.pivot(index="model", columns="lead", values="acc")[[3, 6, 9, 12, 15, 18, 24]].round(3).to_string())
    print(diff.round(3).to_string(index=False))


E14_STATES = {"ro_dmi": ("nino34", "d20_wwv", "dmi"), "ro_dmi_atl3": ("nino34", "d20_wwv", "dmi", "atl3"),
          "ro_dmi_natl": ("nino34", "d20_wwv", "dmi", "nta")}


def e14_atlantic() -> None:
    """E14: does the tropical Atlantic add to the seasonal oscillator with DMI?

    Reads the raw ERSST v6 file for ATL3 and NTA. Writes reports/tables/e14_*.csv."""
    from prepare_data import ERSST_V6

    sst = load_ersst(ERSST_V6, lat_range=(-30.0, 30.0))
    data = load_inputs()["wwv"]
    for region in ("atl3", "nta"):
        data[region] = box_mean(sst, region, weighted=True).reindex(data.index)
    data["dmi"] = data["wtio"] - data["etio"]
    check = (box_mean(sst, "nino34", weighted=True).reindex(data.index) - data["nino34"]).abs().max()
    print("Niño-3.4 check vs dataset:", round(float(check), 6))
    forecasts = pd.concat([cs_lim_forecasts(data, state, name, LEADS, TRAIN_START) for name, state in E14_STATES.items()],
                          ignore_index=True)
    forecasts.to_csv(TABLES / "e14_forecasts.csv", index=False)
    skill = skill_table(forecasts, by=("model", "lead"))
    skill.round(4).to_csv(TABLES / "e14_skill_by_lead.csv", index=False)
    rows = [{"model": m, "lead": lead, **bootstrap_difference(forecasts, m, "ro_dmi", lead)}
            for m in ("ro_dmi_atl3", "ro_dmi_natl") for lead in (3, 6, 9, 12, 15, 18, 24)]
    diff = mark_significant(pd.DataFrame(rows))
    diff.round(4).to_csv(TABLES / "e14_vs_ro_dmi.csv", index=False)
    print(skill.pivot(index="model", columns="lead", values="acc")[[3, 6, 9, 12, 15, 18, 24]].round(3).to_string())
    print(diff.round(3).to_string(index=False))
    t = forecasts[forecasts.lead.isin([6, 9, 12])].groupby(["model", "start_month", "lead"]).apply(
        lambda g: anomaly_correlation(g.forecast, g.observed)).unstack("model")
    for m in ("ro_dmi_atl3", "ro_dmi_natl"):
        print(m, "minus ro_dmi by start month:")
        print((t[m] - t["ro_dmi"]).unstack("lead").round(3).to_string())


# ----------------------------------------------------------------------------
# e15: probabilistic skill
# ----------------------------------------------------------------------------

E15_STATES = {"ro": ("nino34", "d20_wwv"), "ro_dmi": ("nino34", "d20_wwv", "dmi")}


def e15_probabilistic() -> None:
    """E15: probabilistic skill of the seasonal LIM / recharge oscillator.

    Writes reports/tables/e15_*.csv."""
    raw = load_inputs()["wwv"].loc[TRAIN_START:]
    raw["dmi"] = raw["wtio"] - raw["etio"]
    rows = []
    for name, state in E15_STATES.items():
        for year in range(1984, 2027):
            cutoff = pd.Timestamp(f"{year}-01-01")
            anom = anomalies_from_training(raw, cutoff)
            train = anom.loc[: cutoff - pd.DateOffset(days=1), list(state)].dropna()
            x, months = train.to_numpy(), train.index.month.to_numpy()
            mean = x.mean(axis=0)
            xc = x - mean
            G, Q = fit_cyclostationary_lim(xc, months), {}
            for m in range(1, 13):
                idx = np.flatnonzero(months[:-1] == m)
                x0, x1 = xc[idx], xc[idx + 1]
                resid = x1 - x0 @ G[m].T
                Q[m] = resid.T @ resid / max(len(idx) - len(state), 1)
            n34 = anom["nino34"]
            clim_sd = {m: n34[(n34.index < cutoff) & (n34.index.month == m)].std() for m in range(1, 13)}
            for start in anom.index[anom.index.year == year]:
                s0 = anom.loc[start, list(state)].to_numpy(dtype=float)
                if np.isnan(s0).any():
                    continue
                s, P, m = s0 - mean, np.zeros((len(state), len(state))), start.month
                for lead in range(1, max(LEADS) + 1):
                    s, P = G[m] @ s, G[m] @ P @ G[m].T + Q[m]
                    m = m % 12 + 1
                    target = start + pd.DateOffset(months=lead)
                    y = n34.get(target, np.nan)
                    if np.isnan(y):
                        continue
                    mu, sd = s[0] + mean[0], np.sqrt(P[0, 0])
                    lo, hi = norm.ppf([0.1, 0.9], mu, sd)
                    rows.append({"model": name, "origin_year": year, "start": start, "lead": lead, "mu": mu, "sd": sd,
                                 "observed": y, "crps": crps_gaussian(mu, sd, y),
                                 "crps_clim": crps_gaussian(0.0, clim_sd[target.month], y),
                                 "covered": lo <= y <= hi, "sq_err": (y - mu) ** 2})
    d = pd.DataFrame(rows)
    d.to_csv(TABLES / "e15_forecasts.csv", index=False)
    g = d.groupby(["model", "lead"])
    table = pd.DataFrame({"crpss": 1 - g.crps.mean() / g.crps_clim.mean(), "coverage80": g.covered.mean(),
                          "spread_error": g.sd.apply(lambda s: np.mean(s ** 2)) / g.sq_err.mean()}).reset_index()
    table.round(4).to_csv(TABLES / "e15_scores.csv", index=False)
    for metric in ("crpss", "coverage80", "spread_error"):
        print(metric)
        print(table.pivot(index="model", columns="lead", values=metric)[[1, 3, 6, 9, 12, 18, 24]].round(2).to_string())


# ----------------------------------------------------------------------------
# e17: XRO-style quadratic terms
# ----------------------------------------------------------------------------

E17_STATE = ("nino34", "d20_wwv", "dmi")
E17_TERMS = {"ro_dmi": (), "ro_dmi_quad": ("TT",), "ro_dmi_quad_th": ("TT", "Th")}


def quadratic_terms(x: np.ndarray, terms) -> np.ndarray:
    cols = [x[:, 0] ** 2 if t == "TT" else x[:, 0] * x[:, 1] for t in terms]
    return np.column_stack(cols) if cols else np.zeros((len(x), 0))


def fit_quadratic(x: np.ndarray, months: np.ndarray, terms):
    """Per month: linear operator G_m (all equations) and quadratic coefficients for the Niño-3.4 equation."""
    params, linear = {}, fit_cyclostationary_lim(x, months)
    for m in range(1, 13):
        idx = np.flatnonzero(months[:-1] == m)
        x0, x1 = x[idx], x[idx + 1]
        G = linear[m]
        q = np.zeros(len(terms))
        if terms:
            A = np.column_stack([x0, quadratic_terms(x0, terms)])
            coef, *_ = np.linalg.lstsq(A, x1[:, 0], rcond=None)
            G = G.copy()
            G[0] = coef[: x.shape[1]]
            q = coef[x.shape[1]:]
        params[m] = (G, q)
    return params


def e17_quadratic() -> None:
    """E17: XRO-style quadratic terms in the seasonal oscillator.

    Writes reports/tables/e17_*.csv."""
    raw = load_inputs()["wwv"].loc[TRAIN_START:]
    raw["dmi"] = raw["wtio"] - raw["etio"]
    rows = []
    for year in range(1984, 2027):
        cutoff = pd.Timestamp(f"{year}-01-01")
        anom = anomalies_from_training(raw, cutoff)
        train = anom.loc[: cutoff - pd.DateOffset(days=1), list(E17_STATE)].dropna()
        mean = train.mean().to_numpy()
        x, months = train.to_numpy() - mean, train.index.month.to_numpy()
        n34 = anom["nino34"]
        for name, terms in E17_TERMS.items():
            params = fit_quadratic(x, months, terms)
            for start in anom.index[anom.index.year == year]:
                s = anom.loc[start, list(E17_STATE)].to_numpy(dtype=float)
                if np.isnan(s).any():
                    continue
                s, m = s - mean, start.month
                for lead in range(1, max(LEADS) + 1):
                    G, q = params[m]
                    new = G @ s
                    if terms:
                        new[0] += quadratic_terms(s[None, :], terms)[0] @ q
                    s, m = new, m % 12 + 1  # quadratic runs are checked for blow-up below
                    target = start + pd.DateOffset(months=lead)
                    obs = n34.get(target, np.nan)
                    if not np.isnan(obs) and np.isfinite(s[0]):
                        rows.append({"model": name, "origin_year": year, "start": start, "start_month": start.month,
                                     "lead": lead, "forecast": s[0] + mean[0], "observed": obs})
    f = pd.DataFrame(rows)
    f.to_csv(TABLES / "e17_forecasts.csv", index=False)
    skill = skill_table(f, by=("model", "lead"))
    skill.round(4).to_csv(TABLES / "e17_skill_by_lead.csv", index=False)
    diff = mark_significant(pd.DataFrame([{"model": m, "lead": lead, **bootstrap_difference(f, m, "ro_dmi", lead)}
                                          for m in ("ro_dmi_quad", "ro_dmi_quad_th") for lead in (3, 6, 9, 12, 15, 18, 24)]))
    diff.round(4).to_csv(TABLES / "e17_vs_ro_dmi.csv", index=False)
    print(skill.pivot(index="model", columns="lead", values="acc")[[3, 6, 9, 12, 15, 18, 24]].round(3).to_string())
    print(skill.pivot(index="model", columns="lead", values="rmse")[[3, 6, 9, 12, 18, 24]].round(3).to_string())
    print(diff.round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# e18: exact attribution to the initial Niño-3.4 and WWV
# ----------------------------------------------------------------------------

E18_STATE = ["nino34", "d20_wwv"]


def e18_attribution() -> None:
    """E18: exact attribution of the seasonal recharge oscillator's forecasts.

    The linear forecast at lead k is A_k x0, so the Niño-3.4 forecast splits exactly into a
    part from the initial Niño-3.4 and a part from the initial WWV. Same rolling origin and
    training-only fitting as E7. Reports the ACC of each part alone and the share of
    forecast variance from each, by lead and for spring vs other starts.
    Writes reports/tables/e18_attribution.csv."""
    raw = load_inputs()["wwv"].loc[TRAIN_START:]
    rows = []
    for year in range(1984, 2027):
        cutoff = pd.Timestamp(f"{year}-01-01")
        anom = anomalies_from_training(raw, cutoff)
        train = anom.loc[: cutoff - pd.DateOffset(days=1), E18_STATE].dropna()
        mean = train.mean().to_numpy()
        ops = fit_cyclostationary_lim(train.to_numpy() - mean, train.index.month.to_numpy())
        n34 = anom["nino34"]
        for start in anom.index[anom.index.year == year]:
            x0 = anom.loc[start, E18_STATE].to_numpy(dtype=float)
            if np.isnan(x0).any():
                continue
            x0 = x0 - mean
            A, m = np.eye(2), start.month
            for lead in range(1, 25):
                A, m = ops[m] @ A, m % 12 + 1
                obs = n34.get(start + pd.DateOffset(months=lead), np.nan)
                if np.isnan(obs):
                    continue
                rows.append({"start": start, "start_month": start.month, "lead": lead, "observed": obs,
                             "from_nino34": A[0, 0] * x0[0], "from_wwv": A[0, 1] * x0[1], "mean": mean[0]})
    d = pd.DataFrame(rows)
    d["total"] = d.from_nino34 + d.from_wwv + d["mean"]

    def summarise(g):
        acc = lambda f: anomaly_correlation(f, g.observed)  # noqa: E731
        var_t, var_h = g.from_nino34.var(), g.from_wwv.var()
        return pd.Series({"acc_total": acc(g.total), "acc_nino34_part": acc(g.from_nino34), "acc_wwv_part": acc(g.from_wwv),
                          "wwv_variance_share": var_h / (var_t + var_h)})

    d["season"] = np.where(d.start_month.isin([2, 3, 4, 5]), "spring", "other")
    out = pd.concat([d.groupby("lead").apply(summarise).assign(season="all").reset_index(),
                     d.groupby(["season", "lead"]).apply(summarise).reset_index()], ignore_index=True)
    out.round(4).to_csv(TABLES / "e18_attribution.csv", index=False)
    show = out[out.lead.isin([3, 6, 9, 12, 18, 24])]
    print(show.round(2).to_string(index=False))


# ----------------------------------------------------------------------------
# e19-ablation: skill lost without WWV
# ----------------------------------------------------------------------------

ABLATION_LEADS = (3, 6, 9, 12)


def e19_wwv_ablation() -> None:
    """E19 ablation: how much skill is lost when the warm water volume (WWV) is removed from the inputs?

    Compares the seasonal oscillator with and without WWV (Niño-3.4 only), and the strict
    single-task index network with the network trained without WWV (networks.py mtl
    --variant=e19_no_wwv), at 3, 6, 9 and 12 months with the block bootstrap. Reads saved E3,
    E7 and E10 forecasts; writes reports/tables/e19_oscillator_no_wwv.csv and e19_ablation.csv."""
    def read(name):
        return pd.read_csv(TABLES / name, parse_dates=["start"])

    no_wwv = cs_lim_forecasts(load_inputs()["wwv"], ("nino34",), "oscillator_no_wwv", tuple(range(1, 13)), TRAIN_START)
    no_wwv[FORECAST_COLUMNS].to_csv(TABLES / "e19_oscillator_no_wwv.csv", index=False)
    d = pd.concat([select_forecasts(read("e3_e10_strict_forecasts.csv"), "model", {"st": "net"}),
                   select_forecasts(read("e3_e19_no_wwv_forecasts.csv"), "model", {"st": "net_no_wwv"}),
                   select_forecasts(read("e7_forecasts.csv"), "label", {"cs_lim|ro": "osc"}),
                   no_wwv.assign(model="osc_no_wwv")[FORECAST_COLUMNS]], ignore_index=True)
    rows = [{"comparison": f"{a} - {b}", "lead": lead, **bootstrap_difference(d, a, b, lead)}
            for lead in ABLATION_LEADS for a, b in (("net_no_wwv", "net"), ("osc_no_wwv", "osc"))]
    table = mark_significant(pd.DataFrame(rows))
    table.round(4).to_csv(TABLES / "e19_ablation.csv", index=False)
    print(table.round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# operators: the fitted one-month operators by calendar month
# ----------------------------------------------------------------------------


def monthly_operators(x: np.ndarray, months: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """G_m (12 x 2 x 2) from standardised states, using the given month-to-month pair starts."""
    G = fit_cyclostationary_lim(x, months, pairs)
    return np.stack([G[m] for m in range(1, 13)])


def oscillator_operators() -> None:
    """Interpretation of the seasonal recharge oscillator: its one-month operators G_m by calendar month.

    Descriptive: fitted on all months 1958-2025 (no forecasting here), with a 3-year block
    bootstrap over years for the uncertainty (resampled years not cut to the record length).
    Shows how strongly WWV feeds Niño-3.4 and how much of each variable carries over from one
    month to the next (self-coupling) in each calendar month. Writes reports/figures/ro_operator.png and reports/tables/ro_operator.csv."""
    raw = load_inputs()["wwv"].loc[TRAIN_START:"2025-12-01", ["nino34", "d20_wwv"]]
    anom = anomalies_from_training(raw, pd.Timestamp("2026-01-01")).dropna()
    x = ((anom - anom.mean()) / anom.std()).to_numpy()
    months = anom.index.month.to_numpy()
    years = anom.index.year.to_numpy()
    pairs = np.arange(len(x) - 1)
    G = monthly_operators(x, months, pairs)
    rng = np.random.default_rng(0)
    uniq = np.unique(years)
    draws = []
    for _ in range(1000):
        chosen = uniq[block_resample(len(uniq), 3, rng, trim=False)]
        sel = np.concatenate([pairs[years[pairs] == y] for y in chosen])
        draws.append(monthly_operators(x, months, sel))
    draws = np.array(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    terms = {"nino34 self-coupling": (0, 0), "WWV -> nino34": (0, 1), "nino34 -> WWV": (1, 0), "WWV self-coupling": (1, 1)}
    table = pd.DataFrame([{"month": MONTHS[m], "term": name, "value": G[m][i, j], "ci_low": lo[m][i, j], "ci_high": hi[m][i, j]}
                          for name, (i, j) in terms.items() for m in range(12)])
    table.round(4).to_csv(TABLES / "ro_operator.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    colours = {"nino34 self-coupling": ORANGE, "WWV self-coupling": BLUE}
    for name, colour in colours.items():
        part = table[table.term == name]
        axes[0].plot(range(12), part.value, color=colour, linewidth=2, marker="o", markersize=3.5, label=name)
        axes[0].fill_between(range(12), part.ci_low, part.ci_high, color=colour, alpha=0.15, linewidth=0)
    style(axes[0], "Month-to-month self-coupling (standardised)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    colours = {"WWV -> nino34": GREEN, "nino34 -> WWV": PINK}
    for name, colour in colours.items():
        part = table[table.term == name]
        axes[1].plot(range(12), part.value, color=colour, linewidth=2, marker="o", markersize=3.5,
                     label={"WWV -> nino34": "Recharge: WWV → Niño-3.4", "nino34 -> WWV": "Discharge: Niño-3.4 → WWV"}[name])
        axes[1].fill_between(range(12), part.ci_low, part.ci_high, color=colour, alpha=0.15, linewidth=0)
    axes[1].axhline(0, color=AXIS, linewidth=0.8)
    style(axes[1], "Cross-coupling per month (standardised)")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2)
    for ax in axes:
        ax.set_xticks(range(12), MONTHS)
        ax.set_xlabel("Start month of the one-month step", color=INK2, fontsize=9)
    suptitle(fig, "Seasonal recharge oscillator: one-month operators by calendar month, 1958–2025 (95% block-bootstrap bands)")
    fig.tight_layout()
    fig.savefig(FIGURES / "ro_operator.png", dpi=150)
    plt.close(fig)
    print(table.pivot(index="month", columns="term", values="value").reindex(MONTHS).round(2).to_string())


# ----------------------------------------------------------------------------
# e21: warm water volume from in-situ observations (NOAA PMEL) instead of ORAS5
# ----------------------------------------------------------------------------

PMEL_WWV = ROOT / "data" / "external" / "pmel_wwv" / "wwv.dat"


def load_pmel_wwv() -> pd.Series:
    """NOAA PMEL warm water volume (5S-5N, 120E-80W), monthly from January 1980, in units of 1e14 m^3."""
    rows = [line.split() for line in PMEL_WWV.read_text().splitlines()]
    rows = [r for r in rows if len(r) == 3 and r[0].isdigit() and len(r[0]) == 6]
    index = pd.to_datetime([r[0] for r in rows], format="%Y%m")
    return pd.Series([float(r[1]) / 1e14 for r in rows], index=index, name="pmel_wwv")


def e21_pmel_wwv() -> None:
    """E21: the oscillator and ridge regression with PMEL instead of ORAS5 warm water volume.

    Both sources train from January 1980 and are tested on 2000-2026, leads 1-24. Writes
    reports/tables/e21_forecasts.csv, e21_skill_by_lead.csv and e21_comparisons.csv."""
    data = load_inputs()["wwv"][["nino34", "d20_wwv"]].join(load_pmel_wwv()).loc["1980-01-01":]
    print("correlation of the two WWV series, 1980-2025 monthly values:",
          round(float(data["d20_wwv"].corr(data["pmel_wwv"])), 3))
    runs = []
    for source, column in (("oras5", "d20_wwv"), ("pmel", "pmel_wwv")):
        frame = data[["nino34", column]].rename(columns={column: "wwv"})
        config = RollingConfig(first_test_year=2000, last_test_year=2026, leads=LEADS, train_start="1980-01-01",
                               lim_state=("nino34", "wwv"))
        out = rolling_forecasts(frame, FeatureSpec(lags={"nino34": 12, "wwv": 6}), config, [RidgeModel()])
        out = out[out.model.isin(["cs_lim", "ridge"])]
        runs.append(out.assign(model=out.model.map({"cs_lim": "oscillator", "ridge": "ridge"}) + "|" + source))
    forecasts = pd.concat(runs, ignore_index=True)
    forecasts.to_csv(TABLES / "e21_forecasts.csv", index=False)
    skill = skill_table(forecasts, by=("model", "lead"))
    skill.round(4).to_csv(TABLES / "e21_skill_by_lead.csv", index=False)
    pairs = [("oscillator|pmel", "ridge|pmel"), ("oscillator|pmel", "oscillator|oras5"), ("oscillator|oras5", "ridge|oras5")]
    rows = [{"model": a, "vs": b, "lead": lead, **bootstrap_difference(forecasts, a, b, lead)}
            for a, b in pairs for lead in (3, 6, 9, 12, 18, 24)]
    diff = mark_significant(pd.DataFrame(rows))
    diff.round(4).to_csv(TABLES / "e21_comparisons.csv", index=False)
    print(skill.pivot(index="model", columns="lead", values="acc")[[3, 6, 9, 12, 18, 24]].round(2).to_string())
    print(diff.round(3).to_string(index=False))


COMMANDS = {
    "e5": e5_autumn_iod,
    "e13": e13_three_variable,
    "e14": e14_atlantic,
    "e15": e15_probabilistic,
    "e17": e17_quadratic,
    "e18": e18_attribution,
    "e19-ablation": e19_wwv_ablation,
    "e21-pmel": e21_pmel_wwv,
    "operators": oscillator_operators,
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
