"""Gridded deep-learning baselines trained on observations only (E8).

Two standard architectures map tropical SST and 20 °C isotherm depth (D20) anomaly
maps to the Niño-3.4 anomaly at leads 1..L:

* ``cnn``: a convolutional network in the style of Ham et al. (2019) on the last
  ``cnn_window`` months, stacked as channels;
* ``convlstm``: a convolutional LSTM over the last ``lstm_window`` months.

The protocol matches the index networks (mtl.py): at each origin the climatology,
scaling and targets use training years only, training targets must precede the
cutoff, and each lead keeps its own early-stopping epoch chosen on the last
training years. Requires PyTorch (run with the dl_env conda environment).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .forecast import anomalies_from_training
from .mtl import training_mask


@dataclass
class GridConfig:
    leads: tuple[int, ...] = tuple(range(1, 25))
    cnn_window: int = 3
    lstm_window: int = 12
    channels: int = 32
    lstm_hidden: int = 16
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 30
    validation_years: int = 5
    seeds: tuple[int, ...] = tuple(range(5))
    refit_every: int = 1  # years per trained model; each model forecasts only after its cutoff
    # E23 training options; the defaults reproduce E8
    head: str = "dense"  # "dense" on the full feature map, or "coarse" on a map pooled to 2 x 9 cells
    optimizer: str = "adam"  # "adam" (L2 penalty) or "adamw" (decoupled weight decay)
    lr_step: int = 0  # halve the learning rate every lr_step epochs (0: constant)
    clip: float = 0.0  # gradient-norm clipping (0: none)
    input_noise: float = 0.0  # s.d. of Gaussian noise added to training inputs
    # E10b only: "test_early_stopping" picks each lead's epoch on the test years' own targets
    shortcut: str | None = None


def grid_anomalies(fields: np.ndarray, months: pd.DatetimeIndex, ocean: np.ndarray, cutoff: pd.Timestamp) -> np.ndarray:
    """Anomalies from the per-cell calendar-month climatology of training months, divided by
    their training standard deviation over ocean cells. Land and missing values become 0."""
    train = months < cutoff
    out = np.zeros_like(fields, dtype="float32")
    month_of = months.month.to_numpy()
    for m in range(1, 13):
        sel = month_of == m
        clim = np.nanmean(fields[sel & train], axis=0)
        out[sel] = fields[sel] - clim
    scale = np.nanstd(out[train][:, ocean])
    out = np.where(ocean & np.isfinite(out), out / scale, 0.0)
    return out.astype("float32")


def windows(stack: np.ndarray, window: int) -> np.ndarray:
    """(T, C, H, W) -> (T, window, C, H, W): the ``window`` months ending at each month
    (rows without a full history are NaN-free zeros and must be excluded by the caller)."""
    T = len(stack)
    out = np.zeros((T, window, *stack.shape[1:]), dtype="float32")
    for k in range(window):
        out[k:, window - 1 - k] = stack[: T - k]
    return out


class CNN(nn.Module):
    """Three convolution layers on the stacked months, then a small dense head (Ham et al. 2019 style)."""

    def __init__(self, in_channels: int, n_leads: int, channels: int, dropout: float, shape, head: str = "dense"):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels, (4, 8), padding="same"), nn.Tanh(), nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, (2, 4), padding="same"), nn.Tanh(), nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, (2, 4), padding="same"), nn.Tanh(),
            *coarse_pool(head),
        )
        with torch.no_grad():
            n = self.features(torch.zeros(1, in_channels, *shape)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n, 32), nn.Tanh(), nn.Linear(32, n_leads))

    def forward(self, x):  # x: (N, window, C, H, W)
        return self.head(self.features(x.flatten(1, 2)))


def coarse_pool(head: str) -> list:
    """Layers between the feature map and the dense head: none, or pooling to 2 x 9 cells."""
    if head == "dense":
        return []
    if head == "coarse":
        return [nn.AdaptiveAvgPool2d((2, 9))]
    raise ValueError(head)


class ConvLSTMCell(nn.Module):
    """One convolutional LSTM step: gates computed by a single convolution over [input, hidden state]."""

    def __init__(self, in_channels: int, hidden: int, kernel=(3, 5)):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(in_channels + hidden, 4 * hidden, kernel, padding="same")

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    """ConvLSTM over the month sequence on a 2x-pooled grid; the final hidden state feeds a dense head."""

    def __init__(self, in_channels: int, n_leads: int, hidden: int, dropout: float, shape, head: str = "dense"):
        super().__init__()
        self.pool = nn.AvgPool2d(2)  # 2° -> 4° to keep the recurrence cheap
        self.cell = ConvLSTMCell(in_channels, hidden)
        h, w = shape[0] // 2, shape[1] // 2
        cells = (h // 2) * (w // 2) if head == "dense" else 2 * 9
        self.head = nn.Sequential(nn.MaxPool2d(2), *coarse_pool(head), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(hidden * cells, 32), nn.Tanh(), nn.Linear(32, n_leads))

    def forward(self, x):  # x: (N, window, C, H, W)
        x = self.pool(x.flatten(0, 1)).unflatten(0, x.shape[:2])
        h = x.new_zeros(x.shape[0], self.cell.hidden, *x.shape[-2:])
        state = (h, h.clone())
        for t in range(x.shape[1]):
            state = self.cell(x[:, t], state)
        return self.head(state[0])


def build_model(kind: str, cfg: GridConfig, n_channels: int, shape) -> nn.Module:
    """The ``cnn`` or ``convlstm`` network for maps of ``shape`` with ``n_channels`` fields per month."""
    if kind == "cnn":
        return CNN(cfg.cnn_window * n_channels, len(cfg.leads), cfg.channels, cfg.dropout, shape, cfg.head)
    if kind == "convlstm":
        return ConvLSTM(n_channels, len(cfg.leads), cfg.lstm_hidden, cfg.dropout, shape, cfg.head)
    raise ValueError(kind)


def train(kind, cfg: GridConfig, X, y, m, seed, device, *, val=None, epochs=None, Xte=None):
    """Mini-batch Adam. With ``val`` = (X, y, m): per-lead early stopping, returns (best epoch
    per lead, best validation MSE per lead). Without: trains to the largest of ``epochs`` and
    returns the test predictions taken at each lead's own epoch."""
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    model = build_model(kind, cfg, X.shape[2], X.shape[-2:]).to(device)
    optimizer = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[cfg.optimizer]
    opt = optimizer(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.StepLR(opt, cfg.lr_step, 0.5) if cfg.lr_step else None
    n_leads = len(cfg.leads)
    if val is not None:
        n_epochs, best, best_epoch, wait = cfg.max_epochs, np.full(n_leads, np.inf), np.zeros(n_leads, dtype=int), 0
        mv = val[2].float()
    else:
        epochs = np.broadcast_to(np.asarray(epochs, dtype=int), (n_leads,))
        n_epochs, pred = int(epochs.max()), np.full((len(Xte), n_leads), np.nan)
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch in torch.randperm(len(X), generator=gen).split(cfg.batch_size):
            batch = batch.to(device)
            mb = m[batch]
            if not mb.any():
                continue
            opt.zero_grad()
            xb = X[batch]
            if cfg.input_noise:
                xb = xb + cfg.input_noise * torch.randn_like(xb)
            loss = ((model(xb) - y[batch]) ** 2)[mb].mean()
            loss.backward()
            if cfg.clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
        if schedule is not None:
            schedule.step()
        model.eval()
        with torch.no_grad():
            if val is not None:
                sq = (model(val[0]) - val[1]) ** 2 * mv
                score = (sq.sum(0) / mv.sum(0).clamp(min=1)).cpu().numpy()
                improved = score < best - 1e-6
                best[improved], best_epoch[improved] = score[improved], epoch
                wait = 0 if improved.any() else wait + 1
                if wait >= cfg.patience:
                    break
            else:
                hit = epochs == epoch
                if hit.any():
                    pred[:, hit] = model(Xte).cpu().numpy()[:, hit]
    if val is not None:
        return best_epoch, best
    return pred


def rolling_gridded(grids: dict, index_raw: pd.DataFrame, kinds, cfg: GridConfig, first_year: int, last_year: int,
                    device: str = "cpu", log=print, checkpoint_dir=None) -> pd.DataFrame:
    """Rolling-origin forecasts. ``grids``: months, ocean and the raw fields (sst, d20);
    ``index_raw``: raw monthly Niño-3.4 (column ``nino34``) on the same months, for targets.
    One model per ``cfg.refit_every`` origin years, trained on months before the block."""
    months = pd.DatetimeIndex(pd.to_datetime(grids["months"]))
    ocean = grids["ocean"]
    raw = index_raw.reindex(months)
    rows = []
    for block_start in range(first_year, last_year + 1, cfg.refit_every):
        block = range(block_start, min(block_start + cfg.refit_every, last_year + 1))
        if checkpoint_dir is not None and all((checkpoint_dir / f"year_{y}.csv").exists() for y in block):
            log(f"{block_start}: already done")
            continue
        cutoff = pd.Timestamp(f"{block_start}-01-01")
        val_cut = pd.Timestamp(f"{block_start - cfg.validation_years}-01-01")
        stack = np.stack([grid_anomalies(grids[name], months, ocean, cutoff) for name in ("sst", "d20")], axis=1)
        target = anomalies_from_training(raw[["nino34"]], cutoff)["nino34"]
        Y = np.column_stack([target.shift(-lead).to_numpy() for lead in cfg.leads])
        mask_full = training_mask(months, cfg.leads, cutoff) & np.isfinite(Y)
        mask_fit = training_mask(months, cfg.leads, val_cut) & np.isfinite(Y)
        Yt = torch.tensor(np.nan_to_num(Y), dtype=torch.float32, device=device)
        test_rows = np.flatnonzero(months.year.isin(list(block)))
        block_rows = []
        for kind in kinds:
            window = cfg.cnn_window if kind == "cnn" else cfg.lstm_window
            full_history = np.arange(len(months)) >= window - 1
            Xall = torch.tensor(windows(stack, window), device=device)
            fit = np.flatnonzero(full_history & (months < val_cut))
            val = np.flatnonzero(full_history & (months >= val_cut) & (months < cutoff))
            train_rows = np.flatnonzero(full_history & (months < cutoff))

            def part(rows, mask):
                return Xall[rows], Yt[rows], torch.tensor(mask[rows], device=device)

            fit_part, val_part = part(fit, mask_fit), part(val, mask_full)
            if cfg.shortcut == "test_early_stopping":
                fit_part, val_part = part(train_rows, mask_full), part(test_rows, np.isfinite(Y))
            preds = []
            for seed in cfg.seeds:
                best_epoch, _ = train(kind, cfg, *fit_part, seed, device, val=val_part)
                preds.append(train(kind, cfg, *part(train_rows, mask_full), seed, device,
                                   epochs=np.maximum(best_epoch, 1), Xte=Xall[test_rows]))
            pred = np.mean(preds, axis=0)
            for j, lead in enumerate(cfg.leads):
                for k, r in enumerate(test_rows):
                    if np.isfinite(Y[r, j]):
                        start = months[r]
                        block_rows.append({"model": kind, "origin_year": start.year, "start": start,
                                           "start_month": start.month, "lead": lead, "forecast": float(pred[k, j]),
                                           "observed": float(Y[r, j])})
        rows += block_rows
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(block_rows)
            for y in block:
                partial = checkpoint_dir / f"year_{y}.tmp"
                frame[frame.origin_year == y].to_csv(partial, index=False)
                partial.replace(checkpoint_dir / f"year_{y}.csv")
        log(f"{block_start}: done ({', '.join(kinds)})")
    return pd.DataFrame(rows)


def random_cv_gridded(grids: dict, index_raw: pd.DataFrame, kind: str, cfg: GridConfig, device: str = "cpu",
                      log=print) -> pd.DataFrame:
    """E10b, shortcut P4: full-record anomalies and a 5-fold random split over start months
    (10% of each training fold as validation). Every start gets one forecast."""
    months = pd.DatetimeIndex(pd.to_datetime(grids["months"]))
    end = months[-1] + pd.DateOffset(months=1)
    stack = np.stack([grid_anomalies(grids[name], months, grids["ocean"], end) for name in ("sst", "d20")], axis=1)
    target = anomalies_from_training(index_raw.reindex(months)[["nino34"]], end)["nino34"]
    Y = np.column_stack([target.shift(-lead).to_numpy() for lead in cfg.leads])
    window = cfg.cnn_window if kind == "cnn" else cfg.lstm_window
    rows = np.flatnonzero((np.arange(len(months)) >= window - 1) & np.isfinite(Y).any(axis=1))
    Xall = torch.tensor(windows(stack, window), device=device)
    Yt = torch.tensor(np.nan_to_num(Y), dtype=torch.float32, device=device)
    mask = torch.tensor(np.isfinite(Y), device=device)
    pred = np.full(Y.shape, np.nan)
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(rows), 5)
    for k, test_rows in enumerate(folds):
        train_rows = np.setdiff1d(rows, test_rows)
        perm = rng.permutation(train_rows)
        val, fit = np.sort(perm[: len(perm) // 10]), np.sort(perm[len(perm) // 10:])
        test_rows = np.sort(test_rows)
        preds = []
        for seed in cfg.seeds:
            best_epoch, _ = train(kind, cfg, Xall[fit], Yt[fit], mask[fit], seed, device,
                                  val=(Xall[val], Yt[val], mask[val]))
            preds.append(train(kind, cfg, Xall[train_rows], Yt[train_rows], mask[train_rows], seed, device,
                               epochs=np.maximum(best_epoch, 1), Xte=Xall[test_rows]))
        pred[test_rows] = np.mean(preds, axis=0)
        log(f"fold {k + 1}/5 done")
    out = []
    for j, lead in enumerate(cfg.leads):
        for r in rows:
            if months[r].year >= 1984 and np.isfinite(Y[r, j]):
                out.append({"model": f"{kind}|P4", "origin_year": months[r].year, "start": months[r],
                            "start_month": months[r].month, "lead": lead, "forecast": float(pred[r, j]),
                            "observed": float(Y[r, j])})
    return pd.DataFrame(out)
