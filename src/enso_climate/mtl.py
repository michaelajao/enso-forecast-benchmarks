"""Matched single-task and multi-task networks for monthly Niño-3.4 forecasts.

Both networks share the same trunk, inputs, data and training budget. The main
head predicts the Niño-3.4 anomaly at leads 1..12. Auxiliary heads, at the same
leads, predict the RONI phase (La Niña / neutral / El Niño), the Dipole Mode Index,
the tropical-mean SST anomaly and the warm water volume (WWV, from the 20 °C
isotherm depth). Which auxiliary heads enter the loss is set by named task sets
(``MTLConfig.aux_sets``); an auxiliary weight of zero gives the single-task twin.

Optionally the Niño-3.4 and WWV heads sit on a backbone: the forecast of a seasonal
(cyclostationary) linear inverse model of the recharge oscillator, so the network
learns a correction to it. With the correction layers starting at zero, epoch 0 is
the backbone itself.

All targets are recomputed at each forecast origin from training years only, and
any target month on or after the origin's cutoff is masked out of the loss.
Requires PyTorch in the active environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from .forecast import FeatureSpec, anomalies_from_training, build_features, fit_cyclostationary_lim, phase_classes


REGRESSION_TASKS = ("nino34", "dmi", "tropics", "wwv")
BACKBONE_HEADS = {"nino34": "nino34", "d20_wwv": "wwv"}  # backbone variable -> head


@dataclass
class MTLConfig:
    leads: tuple[int, ...] = tuple(range(1, 13))
    hidden: int = 32
    dropout: float = 0.2
    lr: float = 3e-3
    weight_decay: float = 1e-3
    max_epochs: int = 600
    patience: int = 60
    validation_years: int = 5
    aux_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    # name -> auxiliary heads in the loss; models are labelled <name>_w<weight> and <name>_auto
    aux_sets: dict[str, tuple[str, ...]] = field(default_factory=lambda: {"mt": ("phase", "dmi", "tropics")})
    scale_aux_targets: bool = True  # divide auxiliary regression targets by their training std
    seeds: tuple[int, ...] = tuple(range(10))
    roni_scaling_start: int = 1950
    linear_skip: bool = True  # each head = linear(inputs) + nonlinear correction(trunk)
    ridge_init: bool = True  # start the linear path at a ridge fit, trunk output heads at zero
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    per_lead_stopping: bool = True  # each lead keeps its own best epoch
    backbone: tuple[str, ...] | None = None  # e.g. ("nino34", "d20_wwv"): seasonal recharge oscillator
    backbone_block_years: int = 5  # cross-fitting blocks for the backbone on training rows
    detrend: bool = False  # remove a training-only linear trend (see anomalies_from_training)
    # E20: the linear path also sees every input times a start-month indicator, so it has a
    # separate weight vector for each start month; the trunk still sees the plain inputs
    month_skip: bool = False
    # inputs that may be missing in training years (e.g. WWV before 1958): their lags are set
    # to 0 after standardisation where missing, and an 'available' flag is added
    optional_inputs: tuple[str, ...] = ()
    # E10 only: deliberately leaky protocols, to measure how much they inflate skill.
    # "full_preprocessing": climatology, scaling and RONI use every month, test years included;
    # "test_early_stopping": each lead's epoch is chosen on the test year's own targets.
    shortcut: str | None = None
    # E19: name -> input-variable prefixes permuted across each test year's starts; the
    # network also forecasts these permuted copies, reported as "<label>|perm_<name>"
    attribution_groups: dict[str, tuple[str, ...]] | None = None
    attribution_repeats: int = 20


def relative_oni(anom: pd.DataFrame, cutoff: pd.Timestamp, start_year: int) -> pd.Series:
    """RONI from training-only anomalies: 3-month means of Niño-3.4 minus the tropical
    mean, rescaled per calendar month to the ONI's standard deviation over training years."""
    oni = anom["nino34"].rolling(3, center=True).mean()
    trop = anom["tropics"].rolling(3, center=True).mean()
    diff = oni - trop
    train = (oni.index < cutoff) & (oni.index.year >= start_year)
    months = oni.index.month
    ratio = oni[train].groupby(months[train]).std() / diff[train].groupby(months[train]).std()
    return diff * ratio.reindex(months).to_numpy()


def make_targets(anom: pd.DataFrame, cutoff: pd.Timestamp, leads, start_year: int) -> dict[str, pd.DataFrame]:
    """Targets at each lead, indexed by start month. Values whose target month is on or
    after the cutoff are left in place here; training masks them (see training_mask)."""
    roni = relative_oni(anom, cutoff, start_year)
    phase = pd.Series(phase_classes(roni), index=roni.index).where(roni.notna())
    series = {"nino34": anom["nino34"], "dmi": anom["wtio"] - anom["etio"], "tropics": anom["tropics"], "phase": phase}
    if "d20_wwv" in anom:
        series["wwv"] = anom["d20_wwv"]
    return {name: pd.DataFrame({lead: s.shift(-lead) for lead in leads}) for name, s in series.items()}


def training_mask(index: pd.DatetimeIndex, leads, cutoff: pd.Timestamp) -> np.ndarray:
    """True where the target month (start + lead) lies before the cutoff."""
    target_months = np.array([[(t + pd.DateOffset(months=lead)) < cutoff for lead in leads] for t in index])
    return target_months


# ----------------------------------------------------------------------------
# Seasonal recharge-oscillator backbone
# ----------------------------------------------------------------------------


def lim_paths(state: np.ndarray, months: np.ndarray, operators, mean: np.ndarray, leads) -> np.ndarray:
    """Cyclostationary LIM forecasts from every start: (n starts, len(leads), n variables)."""
    G = np.stack([np.eye(state.shape[1])] + [operators[m] for m in range(1, 13)])
    x, month = state - mean, months.copy()
    out = np.full((len(state), len(leads), state.shape[1]), np.nan)
    for lead in range(1, max(leads) + 1):
        x = np.einsum("nij,nj->ni", G[month], x)
        month = month % 12 + 1
        if lead in leads:
            out[:, list(leads).index(lead)] = x + mean
    return out


def backbone_forecasts(anom: pd.DataFrame, variables, leads, fit_until: pd.Timestamp, block_years: int) -> np.ndarray:
    """Backbone forecasts for every start month, using data before ``fit_until`` only.

    Starts on or after ``fit_until`` use the LIM fitted on all earlier months. Starts before
    it are cross-fitted: the LIM is refitted without their block of ``block_years`` years, so
    the network learns from residuals that resemble out-of-sample ones."""
    state = anom[list(variables)]
    months = anom.index.month.to_numpy()
    out = np.full((len(anom), len(leads), len(variables)), np.nan)
    ok = state.notna().all(axis=1).to_numpy()
    before = ok & (anom.index < fit_until)
    # month-to-month pairs entirely before fit_until
    pair = before[:-1] & before[1:]

    def fit(use_pairs):
        x = state.to_numpy(dtype=float)
        mean = x[:-1][use_pairs].mean(axis=0)
        return fit_cyclostationary_lim(x - mean, months, np.flatnonzero(use_pairs)), mean

    ops, mean = fit(pair)
    rows = np.flatnonzero(ok & (anom.index >= fit_until))
    if len(rows):
        out[rows] = lim_paths(state.to_numpy(dtype=float)[rows], months[rows], ops, mean, leads)
    years = anom.index.year.to_numpy()
    train_years = np.unique(years[before])
    for first in range(train_years.min(), train_years.max() + 1, block_years):
        block = (years >= first) & (years < first + block_years)
        rows = np.flatnonzero(before & block)
        if not len(rows):
            continue
        ops_b, mean_b = fit(pair & ~block[:-1] & ~block[1:])
        out[rows] = lim_paths(state.to_numpy(dtype=float)[rows], months[rows], ops_b, mean_b, leads)
    return out


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------


class Net(nn.Module):
    """Shared trunk with task heads. With ``linear_skip`` every head also has a direct
    linear map from the inputs, so the network can start from a ridge-like forecast and
    the trunk only adds a nonlinear correction (identical in single- and multi-task).
    A backbone forecast, when given, is added to the matching heads. With ``n_trunk`` only the
    first ``n_trunk`` input columns feed the trunk and the linear path sees them all (E20)."""

    def __init__(self, n_in: int, n_leads: int, hidden: int, dropout: float, linear_skip: bool = False,
                 n_trunk: int | None = None):
        super().__init__()
        self.n_trunk = n_trunk or n_in
        self.skips = nn.ModuleDict({name: nn.Linear(n_in, n_leads * (3 if name == "phase" else 1))
                                    for name in (*REGRESSION_TASKS, "phase")}) if linear_skip else None
        self.trunk = nn.Sequential(nn.Linear(self.n_trunk, hidden), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, n_leads) for name in REGRESSION_TASKS})
        self.heads["phase"] = nn.Linear(hidden, n_leads * 3)
        self.n_leads = n_leads

    def forward(self, x, base=None):
        h = self.trunk(x[:, :self.n_trunk])
        out = {name: head(h) for name, head in self.heads.items()}
        if self.skips is not None:
            out = {name: value + self.skips[name](x) for name, value in out.items()}
        for name, value in (base or {}).items():
            out[name] = out[name] + value
        out["phase"] = out["phase"].view(-1, self.n_leads, 3)
        return out


def masked_losses(out, targets, masks):
    """Mean squared error per regression head and cross-entropy for the phase head, over unmasked entries only."""
    losses = {}
    for name in REGRESSION_TASKS:
        if name not in targets:
            continue
        m = masks[name]
        losses[name] = ((out[name] - targets[name]) ** 2)[m].mean() if m.any() else out[name].sum() * 0
    m = masks["phase"]
    if m.any():
        logits = out["phase"][m]
        losses["phase"] = nn.functional.cross_entropy(logits, targets["phase"][m].long())
    else:
        losses["phase"] = out["phase"].sum() * 0
    return losses


def total_loss(losses, aux_weight, aux_tasks=("phase", "dmi", "tropics")):
    """Niño-3.4 loss plus ``aux_weight`` times the losses of ``aux_tasks``; weight 0 is the single-task twin."""
    if aux_weight == 0:
        return losses["nino34"]
    return losses["nino34"] + aux_weight * sum(losses[name] for name in aux_tasks)


def ridge_weights(Z, Y, M, alphas):
    """Closed-form ridge per lead on standardised inputs. Z (n, p), Y and M (n, leads);
    alphas has one value per lead. Returns weights (leads, p) and intercepts (leads,)."""
    W, b = np.zeros((Y.shape[1], Z.shape[1])), np.zeros(Y.shape[1])
    for j in range(Y.shape[1]):
        rows = M[:, j]
        if not rows.any():
            continue
        x, y = Z[rows], Y[rows, j]
        xm, ym = x.mean(axis=0), y.mean()
        xc = x - xm
        W[j] = np.linalg.solve(xc.T @ xc + alphas[j] * np.eye(Z.shape[1]), xc.T @ (y - ym))
        b[j] = ym - xm @ W[j]
    return W, b


def choose_ridge_alphas(Zfit, Yfit, Mfit, Zval, Yval, Mval, grid):
    """Per-lead ridge penalty chosen on the validation years (all before the cutoff)."""
    errors = []
    for alpha in grid:
        W, b = ridge_weights(Zfit, Yfit, Mfit, np.full(Yfit.shape[1], alpha))
        sq = np.where(Mval, (Zval @ W.T + b - np.nan_to_num(Yval)) ** 2, 0.0)
        errors.append(sq.sum(axis=0) / np.maximum(Mval.sum(axis=0), 1))
    return np.asarray(grid)[np.argmin(errors, axis=0)]


def apply_init(net: Net, init) -> None:
    """Set the linear path of each named head to (W, b) and its trunk output layer to zero."""
    with torch.no_grad():
        for name, (W, b) in init.items():
            net.skips[name].weight.copy_(torch.as_tensor(W, dtype=torch.float32))
            net.skips[name].bias.copy_(torch.as_tensor(b, dtype=torch.float32))
            net.heads[name].weight.zero_()
            net.heads[name].bias.zero_()


def make_batch(Z, targets, mask_rows, base, rows, device):
    """Tensors for the selected rows: inputs, targets, loss masks and backbone forecasts."""
    sel = np.flatnonzero(rows)
    t = {name: torch.tensor(np.nan_to_num(df.to_numpy(dtype=float)[sel]), dtype=torch.float32, device=device)
         for name, df in targets.items()}
    m = {name: torch.tensor(mask_rows[sel] & df.notna().to_numpy()[sel], device=device) for name, df in targets.items()}
    b = {name: torch.tensor(v[sel], dtype=torch.float32, device=device) for name, v in base.items()}
    return {"x": torch.tensor(Z[sel], dtype=torch.float32, device=device), "t": t, "m": m, "base": b}


def train_net(train, aux_weight, aux_tasks, cfg: MTLConfig, seed: int, device, *, val=None, epochs=None, test=None,
              init=None):
    """Full-batch Adam on a batch from ``make_batch``.

    With ``val``: early stopping on the main task's validation MSE, per lead if
    ``cfg.per_lead_stopping`` (else pooled over leads). Returns (best epoch per lead,
    best validation MSE per lead, None).
    Without: trains until the largest of ``epochs`` (one per lead) and returns the main-head
    predictions for ``test`` taken at each lead's own epoch: (epochs, None, predictions).
    With ``init`` the network starts from a known linear forecast, so epoch 0 is a valid choice."""
    torch.manual_seed(seed)
    net = Net(train["x"].shape[1], len(cfg.leads), cfg.hidden, cfg.dropout, cfg.linear_skip,
              train.get("n_trunk")).to(device)
    if init is not None:
        apply_init(net, init)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_leads = len(cfg.leads)
    first = 0 if init is not None else 1
    if val is not None:
        n_epochs = cfg.max_epochs
        best, best_epoch, wait = np.full(n_leads, np.inf), np.zeros(n_leads, dtype=int), 0
        m = val["m"]["nino34"].float()
    else:
        epochs = np.broadcast_to(np.asarray(epochs, dtype=int), (n_leads,))
        n_epochs = int(epochs.max())
        pred = np.full((len(test["x"]), n_leads), np.nan)
    for epoch in range(first, n_epochs + 1):
        if epoch > 0:
            net.train()
            opt.zero_grad()
            loss = total_loss(masked_losses(net(train["x"], train["base"]), train["t"], train["m"]), aux_weight, aux_tasks)
            loss.backward()
            opt.step()
        if val is not None:
            net.eval()
            with torch.no_grad():
                sq = (net(val["x"], val["base"])["nino34"] - val["t"]["nino34"]) ** 2 * m
            if cfg.per_lead_stopping:
                score = (sq.sum(0) / m.sum(0).clamp(min=1)).cpu().numpy()
            else:
                score = np.full(n_leads, (sq.sum() / m.sum()).item())
            improved = score < best - 1e-6
            best[improved], best_epoch[improved] = score[improved], epoch
            wait = 0 if improved.any() else wait + 1
            if wait >= cfg.patience:
                break
        else:
            hit = epochs == epoch
            if hit.any():
                net.eval()
                with torch.no_grad():
                    pred[:, hit] = net(test["x"], test["base"])["nino34"].cpu().numpy()[:, hit]
    if val is not None:
        return best_epoch, best, None
    return epochs, None, pred


def rolling_mtl(raw: pd.DataFrame, spec: FeatureSpec, cfg: MTLConfig, first_year: int, last_year: int,
                train_start: str = "1955-01-01", device: str | None = None, log=print,
                checkpoint_dir=None) -> pd.DataFrame:
    """Rolling-origin forecasts for every auxiliary task set and weight (0 = the shared
    single-task twin), averaged over seeds, plus an '<set>_auto' variant per task set whose
    weight is chosen on the last training years.

    With ``checkpoint_dir``, each origin year is written to <dir>/year_<Y>.csv as soon as it
    finishes and years already on disk are skipped, so an interrupted run can resume."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    raw = raw.loc[train_start:]
    rows = []
    for year in range(first_year, last_year + 1):
        if checkpoint_dir is not None and (checkpoint_dir / f"year_{year}.csv").exists():
            log(f"{year}: already done")
            continue
        year_start = len(rows)
        cutoff = pd.Timestamp(f"{year}-01-01")
        val_cut = pd.Timestamp(f"{year - cfg.validation_years}-01-01")
        pre_cutoff = raw.index[-1] + pd.DateOffset(months=1) if cfg.shortcut == "full_preprocessing" else cutoff
        anom = anomalies_from_training(raw, pre_cutoff, detrend=cfg.detrend)
        X = build_features(anom, spec)
        targets = make_targets(anom, pre_cutoff, cfg.leads, cfg.roni_scaling_start)
        optional = [c for c in X.columns if any(c.startswith(f"{v}_lag") for v in cfg.optional_inputs)]
        if optional:
            X["available"] = X[optional].notna().all(axis=1).astype(float)
        complete = X.drop(columns=optional).notna().all(axis=1).to_numpy()
        idx = anom.index
        train_rows = complete & (idx < cutoff)
        scale_rows = complete if cfg.shortcut == "full_preprocessing" else train_rows
        mean, std = X[scale_rows].mean(), X[scale_rows].std().replace(0, 1).fillna(1)
        Z = (X - mean) / std
        Z[optional] = Z[optional].fillna(0.0)
        Z = Z.to_numpy(dtype=float)
        n_trunk = None
        if cfg.month_skip:
            n_trunk = Z.shape[1]
            onehot = idx.month.to_numpy()[:, None] == np.arange(1, 13)[None, :]
            Z = np.hstack([Z, (Z[:, :, None] * onehot[:, None, :]).reshape(len(Z), -1)])
        mask_full = training_mask(idx, cfg.leads, cutoff)
        mask_fit = training_mask(idx, cfg.leads, val_cut)

        scales = {}
        if cfg.scale_aux_targets:
            for name in REGRESSION_TASKS[1:]:
                if name in targets:
                    v = targets[name].to_numpy(dtype=float)[train_rows][mask_full[train_rows]]
                    scales[name] = float(np.nanstd(v)) or 1.0
                    targets[name] = targets[name] / scales[name]

        base_fit, base_full = {}, {}
        if cfg.backbone:
            paths_fit = backbone_forecasts(anom, cfg.backbone, cfg.leads, val_cut, cfg.backbone_block_years)
            paths_full = backbone_forecasts(anom, cfg.backbone, cfg.leads, cutoff, cfg.backbone_block_years)
            for k, var in enumerate(cfg.backbone):
                head = BACKBONE_HEADS[var]
                if head in targets:
                    scale = scales.get(head, 1.0)
                    base_fit[head], base_full[head] = paths_fit[:, :, k] / scale, paths_full[:, :, k] / scale
            ok = np.isfinite(paths_full).all(axis=(1, 2)) & np.isfinite(paths_fit).all(axis=(1, 2))
            complete = complete & ok
            train_rows = train_rows & ok

        fit_rows = complete & (idx < val_cut)
        val_rows = complete & (idx >= val_cut) & (idx < cutoff)
        test_rows = complete & (idx.year == year)
        if not test_rows.any():
            continue
        fit = make_batch(Z, targets, mask_fit, base_fit, fit_rows, device)
        val = make_batch(Z, targets, mask_full, base_fit, val_rows, device)  # validation targets precede the cutoff
        full = make_batch(Z, targets, mask_full, base_full, train_rows, device)
        test = make_batch(Z, targets, mask_full, base_full, test_rows, device)
        for batch in (fit, val, full, test):
            batch["n_trunk"] = n_trunk
        perm_names = []
        if cfg.attribution_groups:
            rng = np.random.default_rng(year)
            Zt, copies = Z[test_rows], []
            for name, prefixes in cfg.attribution_groups.items():
                cols = [i for i, c in enumerate(X.columns) if c.startswith(tuple(prefixes))]
                for _ in range(cfg.attribution_repeats):
                    Zp = Zt.copy()
                    Zp[:, cols] = Zt[rng.permutation(len(Zt))][:, cols]
                    copies.append(Zp)
                perm_names.append(name)
            test = dict(test, x=torch.tensor(np.concatenate([Zt, *copies]), dtype=torch.float32, device=device))

        init_fit = init_full = None
        if cfg.ridge_init:
            # heads on the backbone start at the backbone (zero correction); others at ridge
            init_fit, init_full = {}, {}
            for name in (n for n in REGRESSION_TASKS if n in targets):
                if name in base_fit:
                    zero = (np.zeros((len(cfg.leads), Z.shape[1])), np.zeros(len(cfg.leads)))
                    init_fit[name] = init_full[name] = zero
                    continue
                y = targets[name].to_numpy(dtype=float)
                ok_y = ~np.isnan(y)
                f, v, a = np.flatnonzero(fit_rows), np.flatnonzero(val_rows), np.flatnonzero(train_rows)
                alphas = choose_ridge_alphas(Z[f], y[f], mask_fit[f] & ok_y[f], Z[v], y[v], mask_full[v] & ok_y[v],
                                             cfg.ridge_alphas)
                init_fit[name] = ridge_weights(Z[f], np.nan_to_num(y[f]), mask_fit[f] & ok_y[f], alphas)
                init_full[name] = ridge_weights(Z[a], np.nan_to_num(y[a]), mask_full[a] & ok_y[a], alphas)

        if cfg.shortcut == "test_early_stopping":
            fit, init_fit = full, init_full
            val = make_batch(Z, targets, np.ones_like(mask_full), base_full, test_rows, device)

        settings = [("st", 0.0, ())] if 0.0 in cfg.aux_weights else []
        settings += [(f"{name}_w{w:g}", w, tasks) for name, tasks in cfg.aux_sets.items() for w in cfg.aux_weights if w > 0]
        predictions, val_scores = {}, {}
        for label, w, tasks in settings:
            preds, scores = [], []
            for seed in cfg.seeds:
                best_epoch, best_val, _ = train_net(fit, w, tasks, cfg, seed, device, val=val, init=init_fit)
                if init_fit is None:
                    best_epoch = np.maximum(best_epoch, 10)
                _, _, pred = train_net(full, w, tasks, cfg, seed, device, epochs=best_epoch, test=test, init=init_full)
                preds.append(pred)
                scores.append(float(np.mean(best_val)))
            mean_pred = np.mean(preds, axis=0)
            n_test = int(test_rows.sum())
            predictions[label] = (w, mean_pred[:n_test])
            for k, name in enumerate(perm_names):
                block = mean_pred[n_test * (1 + k * cfg.attribution_repeats):n_test * (1 + (k + 1) * cfg.attribution_repeats)]
                predictions[f"{label}|perm_{name}"] = (w, block.reshape(cfg.attribution_repeats, n_test, -1).mean(axis=0))
            val_scores[label] = float(np.mean(scores))
        autos = {}
        for name in cfg.aux_sets:
            candidates = [label for label, w, _ in settings if label.startswith(f"{name}_w")]
            if candidates:
                autos[f"{name}_auto"] = min(candidates, key=lambda label: val_scores[label])
        starts = idx[test_rows]
        outputs = [(label, label) for label in predictions] + list(autos.items())
        for label, source in outputs:
            w, pred = predictions[source]
            for j, lead in enumerate(cfg.leads):
                obs = targets["nino34"][lead][test_rows].to_numpy()
                for s, p, o in zip(starts, pred[:, j], obs):
                    if not np.isnan(o):
                        rows.append({"model": label, "origin_year": year, "start": s, "start_month": s.month,
                                     "lead": lead, "forecast": float(p), "observed": float(o), "aux_weight": w})
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            partial = checkpoint_dir / f"year_{year}.tmp"
            pd.DataFrame(rows[year_start:]).to_csv(partial, index=False)
            partial.replace(checkpoint_dir / f"year_{year}.csv")
        log(f"{year}: done (auto " + ", ".join(f"{k}={v}" for k, v in autos.items()) + "; validation MSE "
            + ", ".join(f"{k}:{v:.3f}" for k, v in val_scores.items()) + ")")
    return pd.DataFrame(rows)
