"""Rolling-origin forecast evaluation with training-only preprocessing.

For each test year Y (the "origin"), every quantity estimated from data uses only
months before 1 January Y: anomaly climatologies, standardisation, regression
coefficients, LIM operators and hyperparameters. A training sample (start month t,
lead L) is used only if its target month t+L is before 1 January Y, so no
training target falls in the test period. Test forecasts start in each month of
year Y and use predictors up to and including the start month.

Quarterly ocean heat content is made available only from the last month of each
calendar quarter onwards (plus an optional publication delay), so a forecast never
uses a quarter that is not yet complete.

The module also holds the predictor tables and named input sets shared by the experiments
(read from data/processed/, raw values; anomalies are computed inside each training window),
untrained SVR and random-forest pipelines, and the verification scores (anomaly correlation,
block bootstrap, Gaussian CRPS and ENSO-phase skill). No PyTorch or netCDF4 is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm, logm
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .paths import PROCESSED_DIR

# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------


def training_climatology(series: pd.Series, cutoff: pd.Timestamp, years: int | None = None) -> pd.Series:
    """Calendar-month mean of ``series`` over months before ``cutoff``.

    ``years`` limits the reference to the last N complete years before the cutoff
    (e.g. 30, like an operational climatology); None uses all training years."""
    train = series[series.index < cutoff].dropna()
    if years is not None:
        train = train[train.index.year >= cutoff.year - years]
    clim = train.groupby(train.index.month).mean()
    if len(clim) != 12:
        raise ValueError(f"{series.name}: training period lacks some calendar months")
    return clim


def anomalies_from_training(raw: pd.DataFrame, cutoff: pd.Timestamp, years: int | None = None,
                            detrend: bool = False) -> pd.DataFrame:
    """Anomalies of every column relative to its training-only monthly climatology.

    With ``detrend``, a linear trend fitted to each column's training-month anomalies is
    then removed from all months (used with long records, where warming would otherwise
    bias early anomalies)."""
    out = {}
    t = (raw.index.year + (raw.index.month - 1) / 12).to_numpy(dtype=float)
    for column in raw.columns:
        clim = training_climatology(raw[column], cutoff, years)
        anom = raw[column] - clim.reindex(raw.index.month).to_numpy()
        if detrend:
            train = (raw.index < cutoff) & anom.notna().to_numpy()
            slope, intercept = np.polyfit(t[train], anom.to_numpy()[train], 1)
            anom = anom - (slope * t + intercept)
        out[column] = anom
    return pd.DataFrame(out, index=raw.index)


def quarterly_to_available(quarterly: pd.Series, monthly_index: pd.DatetimeIndex, delay_months: int = 0) -> pd.Series:
    """Most recent complete quarter available at the end of each month.

    A quarter (e.g. Jan-Mar) becomes available at the end of its last month plus
    ``delay_months``. Earlier months see the previous quarter's value."""
    periods = pd.PeriodIndex(quarterly.index, freq="Q")
    available_from = periods.end_time.to_period("M").to_timestamp() + pd.DateOffset(months=delay_months)
    values = pd.Series(quarterly.to_numpy(), index=pd.DatetimeIndex(available_from)).sort_index()
    return values.reindex(monthly_index, method="ffill")


@dataclass
class FeatureSpec:
    """Which predictors enter the feature vector and with how many lags."""

    target: str = "nino34"
    lags: dict[str, int] = field(default_factory=lambda: {"nino34": 12, "wtio": 3, "etio": 3, "tropics": 3})
    seasonal: bool = True  # sin/cos of the start month


def build_features(anom: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Lagged predictors ``<name>_lag0`` (start month) ... ``<name>_lag<n-1>`` plus the month's sin/cos."""
    columns = {}
    for name, n_lags in spec.lags.items():
        for lag in range(n_lags):
            columns[f"{name}_lag{lag}"] = anom[name].shift(lag)
    features = pd.DataFrame(columns, index=anom.index)
    if spec.seasonal:
        angle = 2 * np.pi * (anom.index.month - 1) / 12
        features["sin_month"] = np.sin(angle)
        features["cos_month"] = np.cos(angle)
    return features


# ----------------------------------------------------------------------------
# Predictor tables and input sets
# ----------------------------------------------------------------------------


TRAIN_START = "1958-01-01"  # first ORAS5 year, so every input set uses the same training years
SST_LAGS = {"nino34": 12, "wtio": 3, "etio": 3, "tropics": 3}
INPUT_SETS = {
    # name: (data frame key, feature lags, LIM state or None)
    "sst": ("sst", SST_LAGS, ("nino34", "wtio", "etio", "tropics")),
    "sst+ohcq": ("ohcq", {**SST_LAGS, "ohc_eqpac": 4}, None),
    "sst+wwv": ("wwv", {**SST_LAGS, "d20_wwv": 6, "d20_wwv_west": 6}, ("nino34", "wtio", "etio", "tropics", "d20_wwv", "d20_wwv_west")),
    "ro": ("wwv", {"nino34": 12, "d20_wwv": 6}, ("nino34", "d20_wwv")),  # linear recharge oscillator
}


def load_monthly(name: str = "enso_monthly.csv", directory: str | Path = PROCESSED_DIR) -> pd.DataFrame:
    """A monthly processed table indexed by month start (column ``date``)."""
    return pd.read_csv(Path(directory) / name, parse_dates=["date"], index_col="date")


def load_quarterly(directory: str | Path = PROCESSED_DIR) -> pd.DataFrame:
    """The quarterly processed table (ocean heat content), indexed by quarter label."""
    return pd.read_csv(Path(directory) / "enso_quarterly.csv", index_col="quarter")


def load_inputs(directory: str | Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Raw (absolute) predictors by input-set key; anomalies are computed inside each training window.

    ``sst``: Niño-3.4, western/eastern tropical Indian Ocean and tropical-mean SST; ``ohcq``: plus
    quarterly heat content, available only once its quarter is complete; ``wwv``: plus monthly
    ORAS5 warm water volume; ``long``: SST indices from 1870 (only if prepare_data.py long-indices was run)."""
    monthly = load_monthly(directory=directory)
    quarterly = load_quarterly(directory)
    sst = monthly[["nino34_sst", "wtio_sst", "etio_sst", "tropics_sst"]].rename(columns=lambda c: c.replace("_sst", ""))
    ohcq = sst.copy()
    ohcq["ohc_eqpac"] = quarterly_to_available(quarterly["ohc700_eqpac"].dropna(), sst.index)
    wwv = sst.join(monthly[["d20_wwv", "d20_wwv_west", "d20_wwv_east"]])
    frames = {"sst": sst, "ohcq": ohcq, "wwv": wwv}
    if (Path(directory) / "enso_monthly_long.csv").exists():  # from prepare_data.py long-indices (E9)
        frames["long"] = load_monthly("enso_monthly_long.csv", directory)
    return frames


# ----------------------------------------------------------------------------
# Models: fit(X, y, start_months) / predict(X, start_months), one per lead
# ----------------------------------------------------------------------------


class Climatology:
    """Forecast of zero anomaly."""

    name = "climatology"

    def fit(self, X, y, months):
        return self

    def predict(self, X, months):
        return np.zeros(len(X))


class RidgeModel:
    """Standardised ridge regression; alpha chosen by chronological CV inside training."""

    name = "ridge"
    alphas = (0.1, 1.0, 10.0, 100.0, 1000.0)

    def fit(self, X, y, months):
        values = X.to_numpy()
        self.mean, self.std = values.mean(0), values.std(0)
        self.std[self.std == 0] = 1.0
        Z = (values - self.mean) / self.std
        best, best_err = None, np.inf
        for alpha in self.alphas:
            errors = []
            for tr, va in TimeSeriesSplit(n_splits=4).split(Z):
                model = Ridge(alpha=alpha).fit(Z[tr], y[tr])
                errors.append(np.mean((model.predict(Z[va]) - y[va]) ** 2))
            if np.mean(errors) < best_err:
                best, best_err = alpha, np.mean(errors)
        self.alpha = best
        self.model = Ridge(alpha=best).fit(Z, y)
        return self

    def predict(self, X, months):
        return self.model.predict((X.to_numpy() - self.mean) / self.std)


class MonthlyRidge:
    """A separate RidgeModel for each start month (E20); each penalty is chosen by
    chronological CV within that month's training samples."""

    name = "seasonal_ridge"

    def fit(self, X, y, months):
        self.models = {m: RidgeModel().fit(X[months == m], y[months == m], months[months == m]) for m in range(1, 13)}
        return self

    def predict(self, X, months):
        out = np.empty(len(X))
        for m in np.unique(months):
            rows = months == m
            out[rows] = self.models[m].predict(X[rows], months[rows])
        return out


class SklearnModel:
    """Wrap a scikit-learn estimator factory; optional hyperparameter grid chosen by
    chronological CV inside the training window only; optional seed averaging."""

    def __init__(self, name, factory, grid=None, seeds=(0,), n_splits=4):
        self.name, self.factory, self.grid, self.seeds, self.n_splits = name, factory, grid or [{}], seeds, n_splits

    def _make(self, params, seed):
        return self.factory(random_state=seed, **params)

    def fit(self, X, y, months):
        values = X.to_numpy()
        self.params = self.grid[0]
        if len(self.grid) > 1:
            best = np.inf
            for params in self.grid:
                errors = []
                for tr, va in TimeSeriesSplit(n_splits=self.n_splits).split(values):
                    model = self._make(params, self.seeds[0]).fit(values[tr], y[tr])
                    errors.append(np.mean((model.predict(values[va]) - y[va]) ** 2))
                if np.mean(errors) < best:
                    best, self.params = np.mean(errors), params
        self.members = [self._make(self.params, seed).fit(values, y) for seed in self.seeds]
        return self

    def predict(self, X, months):
        values = X.to_numpy()
        return np.mean([member.predict(values) for member in self.members], axis=0)


# Untrained SVR and random-forest pipelines. Fit them inside a chronological training fold;
# these factories do not split data, tune models or establish skill.

def build_svr(**kwargs):
    """Median imputation and scaling are fitted together with SVR."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVR(**kwargs))


def build_random_forest(random_state: int = 42, **kwargs):
    """Return a reproducible forest with training-only missing-value imputation."""
    options = {"n_estimators": 300, "n_jobs": -1, "random_state": random_state}
    options.update(kwargs)
    return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(**options))


# ----------------------------------------------------------------------------
# Linear inverse models (fitted on the state vector, forecast all leads at once)
# ----------------------------------------------------------------------------


def fit_lim(state: np.ndarray, tau0: int = 1) -> np.ndarray:
    """Stationary LIM: B = logm(C(tau0) C(0)^-1) / tau0 (Penland & Sardeshmukh 1995)."""
    x0, x1 = state[:-tau0], state[tau0:]
    c0 = x0.T @ x0 / len(x0)
    ct = x1.T @ x0 / len(x0)
    return np.real(logm(ct @ np.linalg.inv(c0))) / tau0


def fit_cyclostationary_lim(state: np.ndarray, months: np.ndarray, pairs: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """One-month propagators G_m with x(t+1) = G_m x(t) for start month m.

    ``state`` (time, variables) must already be centred. ``pairs`` lists the rows t whose step
    t -> t+1 is used (repeats allowed, as in a bootstrap); the default is every consecutive pair."""
    if pairs is None:
        pairs = np.arange(len(state) - 1)
    operators = {}
    for m in range(1, 13):
        idx = pairs[months[pairs] == m]
        x0, x1 = state[idx], state[idx + 1]
        operators[m] = (x1.T @ x0) @ np.linalg.pinv(x0.T @ x0)
    return operators


# ----------------------------------------------------------------------------
# Rolling-origin driver
# ----------------------------------------------------------------------------


@dataclass
class RollingConfig:
    first_test_year: int = 1984
    last_test_year: int = 2026
    leads: tuple[int, ...] = tuple(range(1, 13))
    train_start: str = "1950-01-01"
    climatology_years: int | None = None  # None = all training years
    lim_state: tuple[str, ...] = ("nino34", "wtio", "etio", "tropics")
    detrend: bool = False  # remove a training-only linear trend after the climatology
    full_preprocessing: bool = False  # E10 only: climatology from every month, test years included


def rolling_forecasts(raw: pd.DataFrame, spec: FeatureSpec, config: RollingConfig, models) -> pd.DataFrame:
    """Return one row per (model, origin year, start month, lead) with forecast and observation."""
    raw = raw.loc[config.train_start:]
    rows = []
    for year in range(config.first_test_year, config.last_test_year + 1):
        cutoff = pd.Timestamp(f"{year}-01-01")
        pre_cutoff = raw.index[-1] + pd.DateOffset(months=1) if config.full_preprocessing else cutoff
        anom = anomalies_from_training(raw, pre_cutoff, config.climatology_years, config.detrend)
        X_all = build_features(anom, spec)
        target = anom[spec.target]
        test_starts = anom.index[(anom.index.year == year)]
        for lead in config.leads:
            y_all = target.shift(-lead)
            target_date = anom.index + pd.DateOffset(months=lead)
            ok = X_all.notna().all(axis=1) & y_all.notna()
            train = ok & (target_date < cutoff)
            test = ok & anom.index.isin(test_starts)
            if not test.any():
                continue
            Xtr, ytr, mtr = X_all[train], y_all[train].to_numpy(), anom.index[train].month.to_numpy()
            Xte, yte, mte = X_all[test], y_all[test].to_numpy(), anom.index[test].month.to_numpy()
            for model in models:
                prediction = model.fit(Xtr, ytr, mtr).predict(Xte, mte)
                rows += _rows(model.name, year, anom.index[test], lead, prediction, yte)
        # LIMs use the monthly state up to the cutoff and forecast every lead at once.
        state_train = anom.loc[: cutoff - pd.DateOffset(days=1), list(config.lim_state)].dropna()
        mean = state_train.mean().to_numpy()
        B = fit_lim(state_train.to_numpy() - mean)
        cyclo = fit_cyclostationary_lim(state_train.to_numpy() - mean, state_train.index.month.to_numpy())
        target_pos = list(config.lim_state).index(spec.target)
        for start in test_starts:
            x0 = anom.loc[start, list(config.lim_state)].to_numpy(dtype=float)
            if np.isnan(x0).any():
                continue
            x0 = x0 - mean
            x_cyc = x0.copy()
            month = start.month
            for lead in range(1, max(config.leads) + 1):
                x_cyc = cyclo[month] @ x_cyc
                month = month % 12 + 1
                if lead not in config.leads:
                    continue
                obs = target.get(start + pd.DateOffset(months=lead), np.nan)
                if np.isnan(obs):
                    continue
                stationary = (expm(B * lead) @ x0)[target_pos] + mean[target_pos]
                rows += _rows("lim", year, [start], lead, [stationary], [obs])
                rows += _rows("cs_lim", year, [start], lead, [x_cyc[target_pos] + mean[target_pos]], [obs])
    return pd.DataFrame(rows)


def cs_lim_forecasts(data: pd.DataFrame, state: tuple[str, ...], name: str, leads: tuple[int, ...],
                     train_start: str, **config) -> pd.DataFrame:
    """Rolling-origin forecasts of the seasonal (cyclostationary) LIM on ``state``, labelled ``name``.

    ``config`` is passed to RollingConfig (e.g. ``detrend=True``); no regression models are fitted."""
    config = RollingConfig(leads=leads, train_start=train_start, lim_state=state, **config)
    out = rolling_forecasts(data, FeatureSpec(lags={"nino34": 1}), config, [])
    return out[out.model == "cs_lim"].assign(model=name)


FORECAST_COLUMNS = ["model", "origin_year", "start", "start_month", "lead", "forecast", "observed"]


def _rows(model, year, starts, lead, prediction, observed):
    return [
        {"model": model, "origin_year": year, "start": s, "start_month": s.month, "lead": lead,
         "forecast": float(p), "observed": float(o)}
        for s, p, o in zip(starts, prediction, observed)
    ]


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------


def anomaly_correlation(forecast, observed) -> float:
    """Pearson correlation of forecast and observed anomalies (ACC)."""
    return np.corrcoef(forecast, observed)[0, 1]


def skill_table(forecasts: pd.DataFrame, by: tuple[str, ...] = ("model", "lead")) -> pd.DataFrame:
    """ACC, RMSE, MSE skill score against climatology, amplitude ratio and sample size per group of ``by``."""

    def summarise(group):
        f, o = group["forecast"], group["observed"]
        mse = np.mean((f - o) ** 2)
        return pd.Series({
            "acc": anomaly_correlation(f, o),
            "rmse": np.sqrt(mse),
            "msss": 1 - mse / np.mean(o ** 2),
            "amplitude_ratio": f.std() / o.std(),
            "n": len(group),
        })

    return forecasts.groupby(list(by)).apply(summarise, include_groups=False).reset_index()


def select_forecasts(frame: pd.DataFrame, column: str, names: dict[str, str],
                     columns: list[str] = FORECAST_COLUMNS) -> pd.DataFrame:
    """Rows of ``frame`` whose ``column`` equals each key of ``names``, with ``model`` set to the
    mapped name, restricted to ``columns``; blocks follow the order of ``names``."""
    parts = [frame[frame[column] == old].assign(model=new)[columns] for old, new in names.items()]
    return pd.concat(parts, ignore_index=True)


def mark_significant(diff: pd.DataFrame) -> pd.DataFrame:
    """Add ``significant`` to a table of bootstrap differences: the interval excludes zero."""
    diff["significant"] = (diff.ci_low > 0) | (diff.ci_high < 0)
    return diff


def block_resample(n: int, block: int, rng: np.random.Generator, trim: bool = True) -> np.ndarray:
    """Positions 0..n-1 of one moving-block bootstrap draw: ceil(n / block) runs of ``block``
    consecutive positions with random starts, cut to length n when ``trim``."""
    starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    positions = np.concatenate([np.arange(s, s + block) for s in starts])
    return positions[:n] if trim else positions


def _block_bootstrap_interval(year_of: np.ndarray, statistic, block_years: int, n_boot: int, seed: int) -> np.ndarray:
    """2.5 and 97.5 percentiles of ``statistic(row positions)`` over moving-block resamples of origin years."""
    rng = np.random.default_rng(seed)
    years = np.unique(year_of)
    draws = []
    for _ in range(n_boot):
        chosen = years[block_resample(len(years), block_years, rng)]
        idx = np.concatenate([np.flatnonzero(year_of == y) for y in chosen])
        draws.append(statistic(idx))
    return np.percentile(draws, [2.5, 97.5])


def bootstrap_difference(forecasts: pd.DataFrame, model_a: str, model_b: str, lead: int,
                         metric: str = "acc", block_years: int = 3, n_boot: int = 2000, seed: int = 0) -> dict:
    """Moving-block bootstrap over origin years of skill(model_a) - skill(model_b)."""
    a = forecasts[(forecasts.model == model_a) & (forecasts.lead == lead)].set_index("start")
    b = forecasts[(forecasts.model == model_b) & (forecasts.lead == lead)].set_index("start")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    def score(frame):
        if metric == "acc":
            return anomaly_correlation(frame.forecast, frame.observed)
        return -np.sqrt(np.mean((frame.forecast - frame.observed) ** 2))  # higher is better

    low, high = _block_bootstrap_interval(a.origin_year.to_numpy(), lambda idx: score(a.iloc[idx]) - score(b.iloc[idx]),
                                          block_years, n_boot, seed)
    return {"difference": score(a) - score(b), "ci_low": low, "ci_high": high}


def bootstrap_skill(forecasts: pd.DataFrame, model: str, lead: int, block_years: int = 3, n_boot: int = 2000,
                    seed: int = 0) -> dict:
    """Anomaly correlation of one model at one lead with a moving-block bootstrap interval over origin years."""
    a = forecasts[(forecasts.model == model) & (forecasts.lead == lead)]
    f, o = a.forecast.to_numpy(), a.observed.to_numpy()
    low, high = _block_bootstrap_interval(a.origin_year.to_numpy(), lambda idx: anomaly_correlation(f[idx], o[idx]),
                                          block_years, n_boot, seed)
    return {"acc": anomaly_correlation(f, o), "ci_low": low, "ci_high": high}


# ----------------------------------------------------------------------------
# ENSO-phase and probabilistic scores (phases: 0 La Nina, 1 neutral, 2 El Nino)
# ----------------------------------------------------------------------------


PHASE_THRESHOLD = 0.5  # °C


def crps_gaussian(mu, sigma, y):
    """Continuous ranked probability score of a Gaussian forecast N(mu, sigma^2) for outcome y."""
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def phase_classes(x):
    """ENSO phase of anomalies: 0 at or below -0.5, 2 at or above +0.5, otherwise 1."""
    return np.select([x <= -PHASE_THRESHOLD, x >= PHASE_THRESHOLD], [0, 2], 1)


def phase_scores(forecast, observed) -> dict:
    """Three-class skill of anomaly forecasts: balanced accuracy, Heidke skill score, El Niño and
    La Niña hit rates, and the share of El Niño forecasts that did not verify."""
    f, o = phase_classes(forecast), phase_classes(observed)
    recall = [np.mean(f[o == k] == k) for k in (0, 1, 2)]
    hit = np.mean(f == o)
    chance = sum(np.mean(f == k) * np.mean(o == k) for k in (0, 1, 2))
    false_alarm = np.mean(o[f == 2] != 2) if (f == 2).any() else np.nan
    return {"balanced_accuracy": np.mean(recall), "heidke": (hit - chance) / (1 - chance),
            "el_nino_hit_rate": recall[2], "la_nina_hit_rate": recall[0], "el_nino_false_alarm_ratio": false_alarm}


def recalibrate_amplitude(group: pd.DataFrame, min_years: int = 10) -> pd.Series:
    """Rescale one model's forecasts at one lead to the observed spread, out of sample.

    For origin year Y the scale is std(observed) / std(forecast) over rows from earlier origin
    years only; years with fewer than ``min_years`` earlier origin years are left NaN."""
    out = pd.Series(np.nan, index=group.index)
    for year in sorted(group.origin_year.unique()):
        past = group[group.origin_year < year]
        if past.origin_year.nunique() < min_years:
            continue
        scale = past.observed.std() / past.forecast.std()
        now = group.origin_year == year
        out[now] = group.forecast[now] * scale
    return out
