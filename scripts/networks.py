"""Neural-network experiments (requires PyTorch), rolling origin 1984-2026.

    python scripts/networks.py COMMAND [options]

  mtl        E3/E3b matched multi-task vs single-task index networks, and the E9, E10 and E19 network
             variants: --variant=NAME [--gpu] [--years=A-B] [--collect]
             E24: --variant=e24_ro|e24_backbone [--check-inputs] [--benchmark]
  gridded    E8 CNN and ConvLSTM on SST + D20 maps [--gpu] [--years=A-B] [--collect]; E10b shortcuts
             --p2 [--convlstm] and --p4, and --collect-e10b for the E10b inflation tables
  shortcuts  E10 evaluation shortcuts for ridge, the oscillator and the single-task network [--p6] [--collect]
  tune       E23 tuned CNN and ConvLSTM: --configs, --run=inner|outer [--worker=i/n] [--gpu], --select, --collect

The networks read data/processed/ only. Training is split across processes
with --years and combined with --collect. Writes reports/tables/ and reports/figures/.
"""

from __future__ import annotations

import sys
import time
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enso_climate.paths import ensure_output_directories  # noqa: E402

from enso_climate.forecast import (  # noqa: E402
    FORECAST_COLUMNS, INPUT_SETS, TRAIN_START, FeatureSpec, RidgeModel, RollingConfig, anomalies_from_training,
    anomaly_correlation, bootstrap_difference, bootstrap_skill, build_features, load_inputs, mark_significant,
    rolling_forecasts, select_forecasts, skill_table,
)
from enso_climate.gridded import GridConfig, rolling_gridded  # noqa: E402
from enso_climate.mtl import (  # noqa: E402
    MTLConfig, choose_ridge_alphas, make_batch, make_targets, ridge_weights, rolling_mtl, train_net,
)
from enso_climate.plotting import (  # noqa: E402
    AMBER, AXIS, BLUE, DARK_BLUE, GREEN, INK2, LIGHT_BLUE, LIGHT_GREEN, MUTED, ORANGE, PINK, plot_acc_rmse, style,
    suptitle,
)

TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"


# ----------------------------------------------------------------------------
# mtl: matched multi-task and single-task index networks
# ----------------------------------------------------------------------------

DERIVED = ("phase", "dmi", "tropics")  # labels largely determined by Niño-3.4 itself (plus the IOD)
DYNAMICAL = ("wwv", "dmi")  # other state variables: future recharge and Indian Ocean Dipole
RO_STATE = ("nino34", "d20_wwv")

# Earlier runs: e3_years/ (quarterly heat content, no linear path) and e3_years_v2/ (monthly
# WWV, linear path from a random start, one early-stopping epoch for all leads). The v3
# variants below ran before auxiliary targets were standardised and the WWV head was added
# (scale_aux_targets=False reproduces the loss; the network then had no WWV head).
VARIANTS = {
    "sst_wwv": dict(inputs="sst+wwv", dir="e3_years_v3/sst_wwv", cfg=dict(scale_aux_targets=False)),
    "ro": dict(inputs="ro", dir="e3_years_v3/ro", cfg=dict(scale_aux_targets=False)),
    # E3b: factorial {network alone, on the seasonal recharge-oscillator backbone} x
    # {derived, dynamical auxiliary tasks}
    "ro_tasks": dict(inputs="ro", dir="e3b/ro_tasks",
                     cfg=dict(aux_weights=(0.0, 0.5, 1.0), aux_sets={"mt": DERIVED, "dyn": DYNAMICAL})),
    "ro_backbone": dict(inputs="ro", dir="e3b/ro_backbone",
                        cfg=dict(aux_weights=(0.0, 0.5, 1.0), aux_sets={"mt": DERIVED, "dyn": DYNAMICAL},
                                 backbone=RO_STATE)),
}
# E9: SST indices back to 1880 with WWV as an input that is missing before
# 1958, training-only detrending, development test years 1984-2005 (confirmation 2006-2026).
E9_LAGS = {"nino34": 24, "wtio": 3, "etio": 3, "tropics": 3, "d20_wwv": 6}
E9_CFG = dict(aux_weights=(0.0, 0.5, 1.0), aux_sets={"dyn": DYNAMICAL}, detrend=True, optional_inputs=("d20_wwv",))
E9_REFS = {"recharge_oscillator": "recharge_oscillator", "ridge|sst24|long": "ridge_sst_long"}
for _tag, _start in (("long", "1880-01-01"), ("short", "1958-01-01")):
    for _period, _years in (("dev", (1984, 2005)), ("confirm", (2006, 2026))):
        VARIANTS[f"e9_{_tag}_{_period}"] = dict(frame="long", lags=E9_LAGS, train_start=_start, years=_years,
                                               dir=f"e9/{_tag}", cfg=E9_CFG,
                                               refs=(f"e9_linear_{_period}_forecasts.csv", E9_REFS))
# E10: the strict single-task network (as e3 "ro") rerun with one shortcut.
for _shortcut in (None, "full_preprocessing", "test_early_stopping"):
    VARIANTS[f"e10_{_shortcut or 'strict'}"] = dict(inputs="ro", dir=f"e10/{_shortcut or 'strict'}",
                                        cfg=dict(aux_weights=(0.0,), scale_aux_targets=False, shortcut=_shortcut))
# E19: permutation attribution of the strict single-task network.
VARIANTS["e19_attribution"] = dict(inputs="ro", dir="e19", cfg=dict(
    aux_weights=(0.0,), scale_aux_targets=False,
    attribution_groups={"nino34": ("nino34_lag",), "wwv": ("d20_wwv_lag",), "month": ("sin_month", "cos_month")}))
VARIANTS["e19_no_wwv"] = dict(frame="wwv", lags={"nino34": 12}, dir="e19_no_wwv",
                              cfg=dict(aux_weights=(0.0,), scale_aux_targets=False))
# E20: the strict single-task network with a separate linear map for each start month.
VARIANTS["e20_seasonal"] = dict(inputs="ro", dir="e20/seasonal",
                                cfg=dict(aux_weights=(0.0,), scale_aux_targets=False, month_skip=True))
# E24 is a separate extension; original E3/E3b settings and checkpoints stay unchanged.
E24_LEADS = tuple(range(1, 25))
E24_REFERENCES = {"ridge|ro": "ridge_ro", "cs_lim|ro": "recharge_oscillator"}
VARIANTS["e24_ro"] = dict(inputs="ro", dir="e24_2026-10-29/ro", tag="e24_ro",
                           cfg={**VARIANTS["ro"]["cfg"], "leads": E24_LEADS},
                           refs=("e7_forecasts.csv", E24_REFERENCES))
VARIANTS["e24_backbone"] = dict(inputs="ro", dir="e24_2026-10-29/backbone", tag="e24_backbone",
                                 cfg={**VARIANTS["ro_backbone"]["cfg"], "leads": E24_LEADS,
                                      "aux_weights": (0.0,), "aux_sets": {}},
                                 refs=("e7_forecasts.csv", E24_REFERENCES))
REFERENCES = {"ridge|sst+wwv": "ridge", "ridge|ro": "ridge_ro", "cs_lim|ro": "recharge_oscillator"}
MTL_LABELS = {"st": "Single-task twin", "ridge_sst_long": "Ridge, SST, from 1880 (E9)", "ridge": "Ridge, SST + WWV (E1)", "ridge_ro": "Ridge, Niño-3.4 + WWV (E1)",
          "recharge_oscillator": "Seasonal recharge oscillator (E1)"}
SET_NAMES = {"mt": "derived aux", "dyn": "dynamical aux"}
MTL_COLOURS = {"st": MUTED, "mt_auto": BLUE, "dyn_auto": ORANGE, "ridge": GREEN, "ridge_sst_long": GREEN,
           "ridge_ro": LIGHT_GREEN, "recharge_oscillator": PINK}


def variant_label(model: str) -> str:
    if model in MTL_LABELS:
        return MTL_LABELS[model]
    name, _, rest = model.partition("_")
    setting = "weight chosen in training" if rest == "auto" else f"aux weight {rest[1:]}"
    return f"Multi-task ({SET_NAMES.get(name, name)}), {setting}"


def e24_preflight() -> list[Path]:
    """Report missing training inputs and the primary scoring reference without fitting."""
    paths = [ROOT / "data" / "processed" / name for name in ("enso_monthly.csv", "enso_quarterly.csv")]
    paths.append(TABLES / "e7_forecasts.csv")
    missing = [path for path in paths if not path.is_file()]
    for path in paths:
        print(f"{'MISSING' if path in missing else 'FOUND'} {path}")
    return missing


def ensure_run_manifest(folder: Path, identity: dict) -> None:
    """Refuse to reuse checkpoints from a different configuration, input or code revision."""
    path = folder / "run_manifest.json"
    identity = json.loads(json.dumps(identity, sort_keys=True))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError(f"Run provenance changed; use a new checkpoint folder: {folder}")
        return
    if list(folder.glob("year_*.csv")):
        raise ValueError(f"Checkpoints have no provenance manifest; preserve and audit them first: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    receipt = {"identity": identity, "runtime": {"python": platform.python_version(),
               "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mtl() -> None:
    """E3: matched multi-task vs single-task networks, rolling origin 1984-2026: --variant=NAME [--gpu] [--years=A-B] [--collect].

    Variants are defined explicitly in VARIANTS above. E24 is a separately
    labelled 24-lead extension motivated by earlier results.
    Every variant trains the single-task twin ('st', auxiliary weight 0)
    and the multi-task networks of each auxiliary task set on the same trunk, inputs, seeds
    and per-lead early stopping, with the linear path started from a training-only ridge fit.
    Writes e3_<variant>_*.csv, or the separate e24_ro_/e24_backbone_ prefixes for E24.
    E24 --benchmark trains only 1984-1985 at full settings; --check-inputs fits nothing."""
    name = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--variant=")), "ro_backbone")
    variant = VARIANTS[name]
    extension = name in ("e24_ro", "e24_backbone")
    if extension and "--check-inputs" in sys.argv:
        missing = e24_preflight()
        if missing:
            sys.exit("E24 cannot run/score until the listed artifacts are restored; no training started.")
        return
    if extension and "--collect" not in sys.argv:
        missing = [ROOT / "data" / "processed" / n for n in ("enso_monthly.csv", "enso_quarterly.csv")
                   if not (ROOT / "data" / "processed" / n).is_file()]
        if missing:
            sys.exit("E24 training inputs missing: " + ", ".join(str(p) for p in missing)
                     + ". Restore the processed data; no training started.")
    if "inputs" in variant:
        frame_key, lags, _ = INPUT_SETS[variant["inputs"]]
    else:
        frame_key, lags = variant["frame"], variant["lags"]
    data = load_inputs()[frame_key]
    spec = FeatureSpec(lags=lags)
    train_start = variant.get("train_start", TRAIN_START)
    y0, y1 = variant.get("years", (1984, 2026))
    checkpoints = TABLES / variant["dir"]
    tag = variant.get("tag", f"e3_{name}")
    cfg = MTLConfig(**variant["cfg"])
    if extension and "--collect" not in sys.argv:
        paths = [ROOT / "data" / "processed" / n for n in ("enso_monthly.csv", "enso_quarterly.csv")]
        paths += [ROOT / n for n in ("scripts/networks.py", "src/enso_climate/mtl.py", "src/enso_climate/forecast.py")]
        identity = {"variant": name, "config": asdict(cfg), "train_start": train_start,
                    "test_years": [y0, y1], "sha256": {
                        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
        ensure_run_manifest(checkpoints, identity)
    # Default CPU: these networks are tiny, and CUDA start-up needs a large Windows
    # memory commit that this machine often cannot give. Pass --gpu to use CUDA.
    # --years A-B trains only those origin years (to split work across processes);
    # --collect skips training and analyses all saved years.
    device = "cuda" if "--gpu" in sys.argv else "cpu"
    torch.set_num_threads(2)
    years = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--years=")), f"{y0}-{y1}")
    if extension and "--benchmark" in sys.argv:
        if "--collect" in sys.argv or any(a.startswith("--years=") for a in sys.argv):
            sys.exit("Use --benchmark alone: it trains the fixed first two origins (1984-1985).")
        years = "1984-1985"
    first, last = (int(y) for y in years.split("-"))
    if "--collect" not in sys.argv:
        start = time.time()
        rolling_mtl(data, spec, cfg, first, last, train_start=train_start, device=device, log=lambda m: print(m, flush=True),
                    checkpoint_dir=checkpoints)
        print(f"training time {(time.time() - start) / 60:.1f} min", flush=True)
        if years != f"{y0}-{y1}":
            return
    missing = [y for y in range(y0, y1 + 1) if not (checkpoints / f"year_{y}.csv").exists()]
    if missing:
        sys.exit(f"origin years not yet trained: {missing}")
    forecasts = pd.concat([pd.read_csv(checkpoints / f"year_{y}.csv") for y in range(y0, y1 + 1)], ignore_index=True)
    forecasts.to_csv(TABLES / f"{tag}_forecasts.csv", index=False)

    # E1 references, same protocol and test starts
    forecasts["start"] = pd.to_datetime(forecasts["start"])
    ref_file, ref_labels = variant.get("refs", ("e1_forecasts.csv", REFERENCES))
    refs = select_forecasts(pd.read_csv(TABLES / ref_file, parse_dates=["start"]), "label", ref_labels)
    combined = pd.concat([forecasts.drop(columns="aux_weight"), refs], ignore_index=True)
    skill = skill_table(combined, by=("model", "lead"))
    skill.round(4).to_csv(TABLES / f"{tag}_skill_by_lead.csv", index=False)
    skill_table(combined, by=("model", "start_month", "lead")).round(4).to_csv(TABLES / f"{tag}_skill_by_start_month.csv", index=False)

    # every model against the single-task twin; with two task sets, dynamical against derived
    pairs = [(model, "st") for model in combined.model.unique() if model != "st"]
    if "dyn_auto" in set(combined.model) and "mt_auto" in set(combined.model):
        pairs.append(("dyn_auto", "mt_auto"))
    rows = []
    for model, ref in pairs:
        for lead in cfg.leads:
            for metric in ("acc", "rmse"):
                rows.append({"model": model, "vs": ref, "lead": lead, "metric": metric,
                             **bootstrap_difference(combined, model, ref, lead, metric=metric)})
    diff = mark_significant(pd.DataFrame(rows))
    diff.round(4).to_csv(TABLES / f"{tag}_vs_single_task.csv", index=False)

    shown = [m for m in MTL_COLOURS if m in set(skill.model)]
    inputs = {"sst+wwv": "SST + monthly WWV", "ro": "Niño-3.4 + monthly WWV"}.get(
        variant.get("inputs"), f"SST indices + WWV (missing before 1958), training from {train_start[:4]}")
    backbone = ", on a seasonal recharge-oscillator backbone" if cfg.backbone else ""
    plot_acc_rmse(skill, {m: (variant_label(m), MTL_COLOURS[m], "-") for m in shown}, FIGURES / f"{tag}_skill_by_lead.png",
                  f"Matched multi-task vs single-task networks (10 seeds), {inputs}{backbone}, {y0}–{y1}", key="model")

    print(skill.pivot(index="model", columns="lead", values="acc").round(3).to_string())
    print(skill.pivot(index="model", columns="lead", values="rmse").round(3).to_string())
    acc = diff[(diff.metric == "acc") & diff.lead.isin([3, 6, 9, 12, 18, 24] if extension else [3, 6, 9, 12])]
    print(acc.round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# gridded: CNN and ConvLSTM on SST and D20 maps (E8, E10b)
# ----------------------------------------------------------------------------

CHECKPOINTS = TABLES / "e8_years"
KINDS = ("cnn", "convlstm")
LINES = {
    "cnn": ("CNN on SST + D20 maps (Ham et al. 2019 style)", BLUE),
    "convlstm": ("ConvLSTM on SST + D20 maps", DARK_BLUE),
    "net_ro": ("Index network, Niño-3.4 + WWV (E3)", GREEN),
    "recharge_oscillator": ("Seasonal recharge oscillator (E7)", ORANGE),
    "ridge_ro": ("Ridge, Niño-3.4 + WWV (E7)", AMBER),
}


def origin_years(folder: Path, model: str) -> pd.DataFrame:
    """Concatenate the per-origin-year checkpoints 1984-2026 in ``folder`` under the label ``model``."""
    f = pd.concat([pd.read_csv(folder / f"year_{y}.csv", parse_dates=["start"]) for y in range(1984, 2027)],
                  ignore_index=True)
    return f.assign(model=model)[FORECAST_COLUMNS]


def collect_e10b() -> None:
    """E10b: ACC inflation of the P2 and P4 CNN (and P2 ConvLSTM) over the strict E8 runs, leads 3-24."""
    leads = (3, 6, 9, 12, 18, 24)
    e8 = pd.read_csv(TABLES / "e8_forecasts.csv", parse_dates=["start"])
    p4 = pd.read_csv(TABLES / "e10b_p4_forecasts.csv", parse_dates=["start"])
    allf = pd.concat([e8[e8.model == "cnn"][FORECAST_COLUMNS], origin_years(TABLES / "e10b_p2_years", "cnn|P2"),
                      p4[FORECAST_COLUMNS]], ignore_index=True)
    allf.to_csv(TABLES / "e10b_forecasts.csv", index=False)
    rows = []
    for period, lo, hi in (("1984-2026", 1984, 2026), ("2000-2023", 2000, 2023)):
        part = allf[allf.start.dt.year.between(lo, hi)]
        for shortcut in ("cnn|P2", "cnn|P4"):
            for lead in leads:
                strict = part[(part.model == "cnn") & (part.lead == lead)]
                rows.append({"period": period, "shortcut": shortcut, "lead": lead,
                             "strict_acc": anomaly_correlation(strict.forecast, strict.observed),
                             **bootstrap_difference(part, shortcut, "cnn", lead)})
    table = mark_significant(pd.DataFrame(rows))
    table.round(4).to_csv(TABLES / "e10b_inflation.csv", index=False)

    conv = pd.concat([e8[e8.model == "convlstm"][FORECAST_COLUMNS],
                      origin_years(TABLES / "e10b_p2_convlstm_years", "convlstm|P2")], ignore_index=True)
    rows = []
    for lead in leads:
        acc = {m: anomaly_correlation(g.forecast, g.observed) for m, g in conv[conv.lead == lead].groupby("model")}
        rows.append({"lead": lead, "strict": acc["convlstm"], "P2": acc["convlstm|P2"],
                     **bootstrap_difference(conv, "convlstm|P2", "convlstm", lead)})
    convlstm = pd.DataFrame(rows)
    convlstm.round(4).to_csv(TABLES / "e10b_convlstm_p2.csv", index=False)
    print(table.round(3).to_string(index=False))
    print(convlstm.round(3).to_string(index=False))


def gridded() -> None:
    """E8: gridded deep-learning baselines (CNN, ConvLSTM) trained on observations only [--gpu] [--years=A-B] [--collect].

    Run after prepare_data.py grids (project environment). Inputs: tropical SST (ERSST v6) and
    D20 (ORAS5) anomaly maps, 18S-18N, from 1958. Leads 1-24, test years 1984-2026, compared
    with the E7 linear models and the E3 index networks. Writes reports/tables/e8_*.csv and
    reports/figures/e8_skill_by_lead.png. E10b trains the shortcut runs with
    --p2 [--convlstm] and --p4, and --collect-e10b combines them with E8 into
    reports/tables/e10b_forecasts.csv, e10b_inflation.csv and e10b_convlstm_p2.csv."""
    if "--collect-e10b" in sys.argv:
        collect_e10b()
        return
    cfg = GridConfig()
    device = "cuda" if "--gpu" in sys.argv else "cpu"
    torch.set_num_threads(2)
    checkpoints, kinds = CHECKPOINTS, KINDS
    if "--p2" in sys.argv:  # E10b: test-period early stopping, CNN only
        cfg = GridConfig(shortcut="test_early_stopping")
        checkpoints, kinds = TABLES / "e10b_p2_years", ("cnn",)
        if "--convlstm" in sys.argv:
            checkpoints, kinds = TABLES / "e10b_p2_convlstm_years", ("convlstm",)
    if "--p4" in sys.argv:  # E10b: random monthly split, CNN only
        from enso_climate.gridded import random_cv_gridded

        grids = dict(np.load(ROOT / "data" / "processed" / "tropical_grids.npz"))
        index_raw = load_inputs()["wwv"].loc[TRAIN_START:, ["nino34"]]
        random_cv_gridded(grids, index_raw, "cnn", cfg, device, log=lambda m: print(m, flush=True)).to_csv(
            TABLES / "e10b_p4_forecasts.csv", index=False)
        return
    years = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--years=")), "1984-2026")
    first, last = (int(y) for y in years.split("-"))
    if "--collect" not in sys.argv:
        grids = dict(np.load(ROOT / "data" / "processed" / "tropical_grids.npz"))
        index_raw = load_inputs()["wwv"].loc[TRAIN_START:, ["nino34"]]
        start = time.time()
        rolling_gridded(grids, index_raw, kinds, cfg, first, last, device=device, log=lambda m: print(m, flush=True),
                        checkpoint_dir=checkpoints)
        print(f"training time {(time.time() - start) / 60:.1f} min", flush=True)
        if years != "1984-2026" or checkpoints != CHECKPOINTS:
            return
    missing = [y for y in range(1984, 2027) if not (CHECKPOINTS / f"year_{y}.csv").exists()]
    if missing:
        sys.exit(f"origin years not yet trained: {missing}")
    forecasts = pd.concat([pd.read_csv(CHECKPOINTS / f"year_{y}.csv", parse_dates=["start"]) for y in range(1984, 2027)],
                          ignore_index=True)
    forecasts.to_csv(TABLES / "e8_forecasts.csv", index=False)

    e7 = pd.read_csv(TABLES / "e7_forecasts.csv", parse_dates=["start"])
    refs = [select_forecasts(e7, "label", {"cs_lim|ro": "recharge_oscillator", "ridge|ro": "ridge_ro"})]
    e3 = TABLES / "e3_ro_forecasts.csv"
    if e3.exists():
        refs.append(select_forecasts(pd.read_csv(e3, parse_dates=["start"]), "model", {"st": "net_ro"}))
    combined = pd.concat([forecasts[FORECAST_COLUMNS], *refs], ignore_index=True)
    skill = skill_table(combined, by=("model", "lead"))
    ci = pd.DataFrame([{"model": model, "lead": lead, **bootstrap_skill(combined, model, lead)}
                       for model in KINDS for lead in cfg.leads])
    skill = skill.merge(ci[["model", "lead", "ci_low", "ci_high"]], on=["model", "lead"], how="left")
    skill.round(4).to_csv(TABLES / "e8_skill_by_lead.csv", index=False)
    skill_table(combined, by=("model", "start_month", "lead")).round(4).to_csv(TABLES / "e8_skill_by_start_month.csv", index=False)
    rows = [{"model": model, "vs": ref, "lead": lead, **bootstrap_difference(combined, model, ref, lead)}
            for model in KINDS for ref in ("recharge_oscillator", "ridge_ro") for lead in cfg.leads]
    diff = mark_significant(pd.DataFrame(rows))
    diff.round(4).to_csv(TABLES / "e8_vs_references.csv", index=False)

    plot_acc_rmse(skill, {model: (name, colour, "-") for model, (name, colour) in LINES.items()},
                  FIGURES / "e8_skill_by_lead.png",
                  "Gridded deep-learning baselines trained on observations only vs linear models, 1984–2026",
                  key="model", figsize=(12, 4), xticks=range(1, 25, 2), markersize=3)

    print(skill.pivot(index="model", columns="lead", values="acc").round(2).to_string())
    print(diff[diff.lead.isin([3, 6, 9, 12, 18, 24])].round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# shortcuts: skill inflation from common evaluation shortcuts (E10)
# ----------------------------------------------------------------------------

SHORTCUT_LEADS = tuple(range(1, 13))


def linear_p1(data: pd.DataFrame, lags: dict) -> pd.DataFrame:
    config = RollingConfig(first_test_year=1984, last_test_year=2026, leads=SHORTCUT_LEADS, train_start=TRAIN_START,
                           lim_state=("nino34", "d20_wwv"), full_preprocessing=True)
    out = rolling_forecasts(data, FeatureSpec(lags=lags), config, [RidgeModel()])
    out = out[out.model.isin(["ridge", "cs_lim"])].copy()
    out["model"] = out.model.map({"ridge": "ridge_ro|P1", "cs_lim": "recharge_oscillator|P1"})
    return out


def random_cv(data: pd.DataFrame, lags: dict, device: str) -> pd.DataFrame:
    """P4: anomalies and scaling from the full record, 5-fold random split over start months."""
    raw = data.loc[TRAIN_START:]
    end = raw.index[-1] + pd.DateOffset(months=1)
    anom = anomalies_from_training(raw, end)
    X = build_features(anom, FeatureSpec(lags=lags))
    targets = make_targets(anom, end, SHORTCUT_LEADS, 1950)
    Y = targets["nino34"].to_numpy(dtype=float)
    ok = X.notna().all(axis=1).to_numpy() & np.isfinite(Y).all(axis=1)
    rows_idx = np.flatnonzero(ok)
    Z = ((X - X[ok].mean()) / X[ok].std()).to_numpy(dtype=float)
    cfg = MTLConfig(aux_weights=(0.0,), scale_aux_targets=False)
    ridge_pred = np.full(Y.shape, np.nan)
    net_pred = np.full(Y.shape, np.nan)
    rng = np.random.default_rng(0)
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(rows_idx)):
        tr, te = rows_idx[tr], rows_idx[te]
        ridge_pred[te] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Z[tr], Y[tr]).predict(Z[te])
        # network: random 10% of the training months as validation, as in a shuffled split
        perm = rng.permutation(tr)
        va, fi = perm[: len(tr) // 10], perm[len(tr) // 10:]
        mask = np.ones(Y.shape, dtype=bool)
        # make_batch takes boolean row selections and keeps rows in time order
        sel = {name: np.isin(np.arange(len(Z)), r) for name, r in (("fit", fi), ("val", va), ("full", tr), ("test", te))}
        batch = {name: make_batch(Z, targets, mask, {}, s, device) for name, s in sel.items()}
        alphas = choose_ridge_alphas(Z[fi], Y[fi], mask[fi], Z[va], Y[va], mask[va], cfg.ridge_alphas)
        init_fit = {"nino34": ridge_weights(Z[fi], Y[fi], mask[fi], alphas)}
        init_full = {"nino34": ridge_weights(Z[tr], Y[tr], mask[tr], alphas)}
        preds = []
        for seed in cfg.seeds:
            best_epoch, _, _ = train_net(batch["fit"], 0.0, (), cfg, seed, device, val=batch["val"], init=init_fit)
            _, _, p = train_net(batch["full"], 0.0, (), cfg, seed, device, epochs=best_epoch, test=batch["test"],
                                init=init_full)
            preds.append(p)
        net_pred[np.sort(te)] = np.mean(preds, axis=0)
        print(f"fold {k + 1}/5 done", flush=True)
    out = []
    starts = anom.index
    for name, pred in (("ridge_ro|P4", ridge_pred), ("st|P4", net_pred)):
        for j, lead in enumerate(SHORTCUT_LEADS):
            for r in rows_idx:
                if starts[r].year >= 1984:
                    out.append({"model": name, "origin_year": starts[r].year, "start": starts[r],
                                "start_month": starts[r].month, "lead": lead, "forecast": pred[r, j], "observed": Y[r, j]})
    return pd.DataFrame(out)


def centred_inputs(data: pd.DataFrame, lags: dict, device: str) -> pd.DataFrame:
    """P6: Niño-3.4 input replaced by its centred 3-month mean (ONI-style; includes month t+1).
    Targets keep the monthly Niño-3.4, so ridge and the network are scored on the same target."""
    leaky = data.copy()
    leaky["nino34_c3"] = leaky["nino34"].rolling(3, center=True).mean()
    lags = {"nino34_c3" if k == "nino34" else k: v for k, v in lags.items()}
    config = RollingConfig(first_test_year=1984, last_test_year=2026, leads=SHORTCUT_LEADS, train_start=TRAIN_START)
    ridge = rolling_forecasts(leaky, FeatureSpec(lags=lags), config, [RidgeModel()])
    ridge = ridge[ridge.model == "ridge"].assign(model="ridge_ro|P6")
    cfg = MTLConfig(aux_weights=(0.0,), scale_aux_targets=False)
    net = rolling_mtl(leaky, FeatureSpec(lags=lags), cfg, 1984, 2026, train_start=TRAIN_START, device=device,
                      log=lambda m: print(m, flush=True), checkpoint_dir=TABLES / "e10" / "centred_inputs")
    net = net[net.model == "st"].assign(model="st|P6")
    return pd.concat([ridge[FORECAST_COLUMNS], net[FORECAST_COLUMNS]], ignore_index=True)


def collect_shortcuts() -> None:
    strict = []
    e1 = pd.read_csv(TABLES / "e1_forecasts.csv", parse_dates=["start"])
    strict.append(select_forecasts(e1, "label", {"ridge|ro": "ridge_ro", "cs_lim|ro": "recharge_oscillator"}))
    years = range(1984, 2027)
    for name, label in (("strict", "st"), ("full_preprocessing", "st|P1"), ("test_early_stopping", "st|P2")):
        d = TABLES / "e10" / name
        f = pd.concat([pd.read_csv(d / f"year_{y}.csv", parse_dates=["start"]) for y in years], ignore_index=True)
        strict.append(select_forecasts(f, "model", {"st": label}))
    shortcuts = pd.read_csv(TABLES / "e10_shortcut_forecasts.csv", parse_dates=["start"])
    p6 = TABLES / "e10_p6_forecasts.csv"
    if p6.exists():
        shortcuts = pd.concat([shortcuts, pd.read_csv(p6, parse_dates=["start"])], ignore_index=True)
    allf = pd.concat([*strict, shortcuts[FORECAST_COLUMNS]], ignore_index=True)
    allf.to_csv(TABLES / "e10_all_forecasts.csv", index=False)
    pairs = [("st|P1", "st"), ("st|P2", "st"), ("st|P4", "st"), ("ridge_ro|P1", "ridge_ro"), ("ridge_ro|P4", "ridge_ro"),
             ("recharge_oscillator|P1", "recharge_oscillator"), ("st|P6", "st"), ("ridge_ro|P6", "ridge_ro")]
    rows = []
    for period, lo, hi in (("1984-2026", 1984, 2026), ("2000-2023", 2000, 2023)):
        part = allf[allf.start.dt.year.between(lo, hi)]
        for a, b in pairs:
            for lead in SHORTCUT_LEADS:
                d = bootstrap_difference(part, a, b, lead)
                acc_b = part[(part.model == b) & (part.lead == lead)]
                rows.append({"period": period, "shortcut": a, "strict": b, "lead": lead,
                             "strict_acc": anomaly_correlation(acc_b.forecast, acc_b.observed), **d})
    table = mark_significant(pd.DataFrame(rows))
    table.round(4).to_csv(TABLES / "e10_inflation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    colours = {"st|P1": LIGHT_BLUE, "st|P2": ORANGE, "st|P4": DARK_BLUE, "ridge_ro|P4": GREEN,
               "ridge_ro|P1": LIGHT_GREEN, "recharge_oscillator|P1": PINK, "st|P6": AMBER}
    names = {"st|P1": "Network: full-record preprocessing", "st|P2": "Network: early stopping on test years",
             "st|P4": "Network: random monthly split", "ridge_ro|P4": "Ridge: random monthly split",
             "ridge_ro|P1": "Ridge: full-record preprocessing", "recharge_oscillator|P1": "Oscillator: full-record preprocessing",
             "st|P6": "Network: centred (ONI-style) inputs"}
    for ax, period in zip(axes, ("1984-2026", "2000-2023")):
        for key, colour in colours.items():
            part = table[(table.period == period) & (table.shortcut == key)]
            ax.plot(part.lead, part.difference, color=colour, linewidth=2, marker="o", markersize=3, label=names[key])
        ax.axhline(0, color=AXIS, linewidth=0.8)
        style(ax, f"ACC inflation over the strict protocol, {period}")
        ax.set_xlabel("Lead (months)", color=INK2, fontsize=9)
        ax.set_xticks(range(1, 13))
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    suptitle(fig, "How much common evaluation shortcuts inflate observation-only Niño-3.4 skill")
    fig.tight_layout()
    fig.savefig(FIGURES / "e10_inflation.png", dpi=150)
    plt.close(fig)
    show = table[table.lead.isin([3, 6, 9, 12])]
    print(show[["period", "shortcut", "lead", "strict_acc", "difference", "ci_low", "ci_high", "significant"]]
          .round(3).to_string(index=False))


def shortcuts() -> None:
    """E10: skill inflation from common evaluation shortcuts [--p6] [--collect].

    Run after the E10 network variants (mtl --variant=e10_*).
    * P1 for the linear models: ridge and the seasonal recharge oscillator with
      climatology from every month (test years included).
    * P4: 5-fold random cross-validation over start months (the classic autocorrelation
      leak) for ridge and the single-task network, with full-record preprocessing as such
      studies typically use. Every start 1984-2026 gets one forecast.
    * With --p6: centred three-month (ONI-style) Niño-3.4 inputs for ridge and the network.
    With --collect, combines these with the P1/P2 network runs and reports inflation (ACC
    of the shortcut minus the strict ACC, same starts and leads) with year-block bootstrap
    intervals. Writes reports/tables/e10_*.csv and reports/figures/e10_inflation.png."""
    if "--collect" in sys.argv:
        collect_shortcuts()
        return
    torch.set_num_threads(2)
    device = "cuda" if "--gpu" in sys.argv else "cpu"
    frame_key, lags, _ = INPUT_SETS["ro"]
    data = load_inputs()[frame_key]
    if "--p6" in sys.argv:
        centred_inputs(data, lags, device).to_csv(TABLES / "e10_p6_forecasts.csv", index=False)
        return
    out = pd.concat([linear_p1(data, lags), random_cv(data, lags, device)], ignore_index=True)
    out.to_csv(TABLES / "e10_shortcut_forecasts.csv", index=False)
    print("written", len(out), "rows")


# ----------------------------------------------------------------------------
# tune: CNN and ConvLSTM with hyperparameters chosen on pre-1984 data (E23)
# ----------------------------------------------------------------------------

TUNE = TABLES / "e23"
TUNE_SPACE = {
    "cnn": {"channels": (8, 16, 32), "cnn_window": (3, 6, 12)},
    "convlstm": {"lstm_hidden": (8, 16, 32), "lstm_window": (6, 12, 18)},
}
TUNE_SHARED = {"head": ("dense", "coarse"), "dropout": (0.1, 0.3, 0.5), "weight_decay": (1e-4, 1e-3, 1e-2),
               "lr": (3e-4, 1e-3, 3e-3), "batch_size": (32, 64), "optimizer": ("adam", "adamw"),
               "lr_step": (0, 50), "clip": (0.0, 1.0), "input_noise": (0.0, 0.1)}
TUNE_COUNT = {"cnn": 24, "convlstm": 16}
INNER_YEARS, OUTER_YEARS = (1974, 1983), (1984, 2026)
TUNE_TOP = 5


def tune_configs() -> pd.DataFrame:
    """Configuration 0 of each model is the E8 setting, the rest random draws from the search space."""
    path = TABLES / "e23_configs.csv"
    if path.exists():
        return pd.read_csv(path)
    rng, default, rows = np.random.default_rng(23), GridConfig(), []
    for kind, count in TUNE_COUNT.items():
        space = {**TUNE_SPACE[kind], **TUNE_SHARED}
        drawn = [{k: getattr(default, k) for k in space}]
        while len(drawn) < count:
            draw = {k: v[rng.integers(len(v))] for k, v in space.items()}
            if draw not in drawn:
                drawn.append(draw)
        rows += [{"kind": kind, "config": i, **d} for i, d in enumerate(drawn)]
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    return table


def tune_cfg(row: pd.Series) -> GridConfig:
    """GridConfig of one row of e23_configs.csv (fields of the other model are empty)."""
    fields = {k: row[k] for k in GridConfig.__dataclass_fields__ if k in row and pd.notna(row[k])}
    for k in ("channels", "cnn_window", "lstm_hidden", "lstm_window", "batch_size", "lr_step"):
        if k in fields:
            fields[k] = int(fields[k])
    return GridConfig(**fields)


def tune_jobs(stage: str) -> list[tuple[str, int]]:
    """(model, configuration) pairs of a stage: all for 'inner', the five best of each model for 'outer'."""
    if stage == "inner":
        return list(tune_configs()[["kind", "config"]].itertuples(index=False, name=None))
    selection = pd.read_csv(TABLES / "e23_selection.csv")
    return list(selection[selection["rank"] <= TUNE_TOP].sort_values(["rank", "kind"])[["kind", "config"]]
                .itertuples(index=False, name=None))


def tune_forecasts(stage: str, kind: str, config: int) -> pd.DataFrame:
    first, last = INNER_YEARS if stage == "inner" else OUTER_YEARS
    folder = TUNE / stage / f"{kind}_{config}"
    return pd.concat([pd.read_csv(folder / f"year_{y}.csv", parse_dates=["start"]) for y in range(first, last + 1)],
                     ignore_index=True)


def tune() -> None:
    """E23: CNN and ConvLSTM with hyperparameters and training procedure chosen on pre-1984 data.

    --configs writes reports/tables/e23_configs.csv. --run=inner trains every configuration on test
    years 1974-1983 and --run=outer the five best of each model on 1984-2026, split across processes
    with --worker=i/n or an explicit --only=kind_config,... (checkpoints in reports/tables/e23/, resumable). --select scores the inner runs
    (reports/tables/e23_selection.csv) and --collect writes e23_skill_by_lead.csv and e23_comparisons.csv."""
    configs = tune_configs()
    if "--configs" in sys.argv:
        print(configs.to_string(index=False))
        return
    run = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--run=")), None)
    if run:
        worker, workers = (int(x) for x in
                           next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--worker=")), "0/1").split("/"))
        device = "cuda" if "--gpu" in sys.argv else "cpu"
        torch.set_num_threads(2)
        grids = dict(np.load(ROOT / "data" / "processed" / "tropical_grids.npz"))
        index_raw = load_inputs()["wwv"].loc[TRAIN_START:, ["nino34"]]
        first, last = INNER_YEARS if run == "inner" else OUTER_YEARS
        jobs = tune_jobs(run)[worker::workers]
        only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
        if only:  # explicit job list, e.g. --only=convlstm_3,convlstm_7, for splitting work across machines
            jobs = [(kind, int(config)) for kind, config in (name.rsplit("_", 1) for name in only.split(","))]
        for kind, config in jobs:
            row = configs[(configs.kind == kind) & (configs.config == config)].iloc[0]
            start = time.time()
            rolling_gridded(grids, index_raw, (kind,), tune_cfg(row), first, last, device=device, log=lambda m: None,
                            checkpoint_dir=TUNE / run / f"{kind}_{config}")
            print(f"{run} {kind} {config}: {(time.time() - start) / 60:.1f} min", flush=True)
        return
    if "--select" in sys.argv:
        cutoff = pd.Period(f"{OUTER_YEARS[0]}-01", "M")
        kinds = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--kind=")), ",".join(TUNE_COUNT)).split(",")
        rows = []  # each model's choice depends only on its own configurations, so --kind=cnn can run first
        for kind, config in [j for j in tune_jobs("inner") if j[0] in kinds]:
            f = tune_forecasts("inner", kind, config)
            target = f.start.dt.to_period("M") + f.lead.to_numpy()
            f = f[(target < cutoff).to_numpy() & f.lead.between(3, 24).to_numpy()]
            acc = f.groupby("lead").apply(lambda g: anomaly_correlation(g.forecast, g.observed))
            rows.append({"kind": kind, "config": config, "score": acc.mean(), "acc_6": acc[6], "acc_12": acc[12],
                         "acc_18": acc[18]})
        selection = pd.DataFrame(rows)
        selection["rank"] = selection.groupby("kind").score.rank(ascending=False, method="first").astype(int)
        selection = selection.sort_values(["kind", "rank"])
        selection.round(4).to_csv(TABLES / "e23_selection.csv", index=False)
        print(selection.round(3).to_string(index=False))
        return
    if "--collect" in sys.argv:
        selection = pd.read_csv(TABLES / "e23_selection.csv").set_index(["kind", "config"])["rank"]
        e8 = pd.read_csv(TABLES / "e8_forecasts.csv", parse_dates=["start"])
        e7 = pd.read_csv(TABLES / "e7_forecasts.csv", parse_dates=["start"])
        parts = [select_forecasts(e7, "label", {"cs_lim|ro": "recharge_oscillator"}),
                 e8[FORECAST_COLUMNS].assign(model=e8.model + "_e8")]
        for kind, config in tune_jobs("outer"):
            label = f"{kind}_tuned" if selection[(kind, config)] == 1 else f"{kind}_{config}"
            parts.append(tune_forecasts("outer", kind, config)[FORECAST_COLUMNS].assign(model=label))
        allf = pd.concat(parts, ignore_index=True)
        leads = (3, 6, 9, 12, 18, 24)
        skill = skill_table(allf, by=("model", "lead"))
        for kind in TUNE_COUNT:  # stage 3: best of the top configurations at each lead, chosen on the test years
            pool = skill[skill.model.str.startswith(kind) & ~skill.model.str.endswith("_e8")]
            best = pool.loc[pool.groupby("lead").acc.idxmax(), ["lead", "acc", "model"]]
            skill = pd.concat([skill, best.rename(columns={"model": "best_of"}).assign(model=f"{kind}_test_selected")],
                              ignore_index=True)
        skill.round(4).to_csv(TABLES / "e23_skill_by_lead.csv", index=False)
        rows = [{"model": f"{kind}_tuned", "vs": ref, "lead": lead,
                 **bootstrap_difference(allf, f"{kind}_tuned", ref, lead)}
                for kind in TUNE_COUNT for ref in (f"{kind}_e8", "recharge_oscillator") for lead in leads]
        diff = mark_significant(pd.DataFrame(rows))
        diff.round(4).to_csv(TABLES / "e23_comparisons.csv", index=False)
        print(skill[skill.lead.isin(leads)].pivot(index="model", columns="lead", values="acc").round(2).to_string())
        print(diff.round(3).to_string(index=False))
        return
    sys.exit("give --configs, --run=inner|outer, --select or --collect")


COMMANDS = {"mtl": mtl, "gridded": gridded, "shortcuts": shortcuts, "tune": tune}


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
