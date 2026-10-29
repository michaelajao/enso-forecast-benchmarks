"""Data checks and the processed datasets, from the raw sources in data/raw/ (project environment).

    python scripts/prepare_data.py COMMAND [options]

  inventory      list a configured data source (reports/data_inventory.csv, data_summary.json)
  hash           SHA-256 of every raw file and groups of exact duplicates (reports/tables/raw_*.csv)
  check-grids    stored date, data hash and range of every wind and precipitation grid
  oras5-download ORAS5 20 °C isotherm depth from the Copernicus CDS: FIRST_YEAR [LAST_YEAR] [--process]
  oras5-process  reduce downloaded ORAS5 archives to box means and a 1° band [--delete]
  build          data/processed/enso_monthly.csv and enso_quarterly.csv
  grids          data/processed/tropical_grids.npz (SST and D20 maps for the gridded networks, E8)
  long-indices   data/processed/enso_monthly_long.csv (SST indices from 1870, E9)

Run them in this order on a new machine (oras5-download --process before build). Raw files are
only read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
import warnings
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path

import cdsapi
import netCDF4
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enso_climate.paths import ensure_output_directories  # noqa: E402

from enso_climate.data import (  # noqa: E402
    box_mean, inventory, load_ersst, monthly_anomaly, read_table, relative_anomaly, summarize, write_inventory,
)
from enso_climate.forecast import load_monthly  # noqa: E402
from enso_climate.paths import (  # noqa: E402
    DATA_DIR, REPORTS_DIR, load_local_config, resolve_project_path, source_directory,
)

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
ORAS5_ARCHIVES = ROOT / "data" / "external" / "oras5"  # yearly CDS downloads (not in Git)
ORAS5_DIR = ROOT / "data" / "interim" / "oras5"  # reduced ORAS5 files, from oras5-process


# ----------------------------------------------------------------------------
# inventory: what is in a configured data source
# ----------------------------------------------------------------------------

def inventory_report() -> None:
    """Inventory a configured data source and optionally preview a table."""
    parser = argparse.ArgumentParser(description="Inventory a configured data source and optionally preview a table.")
    parser.add_argument("--source", type=Path, help="Override the data source in the local configuration.")
    parser.add_argument("--config", type=Path, help="Use another TOML configuration file.")
    args = parser.parse_args()
    config = {} if args.source and not args.config else load_local_config(args.config, required=args.config is not None)
    source = args.source or resolve_project_path(config.get("source_path", config.get("archive_path", DATA_DIR / "raw")))
    default_missing = not source.exists() and source.resolve() == (DATA_DIR / "raw").resolve()
    rows = [] if default_missing else inventory(source)
    if not rows and config.get("table_path"):
        rows = inventory(resolve_project_path(config["table_path"]))
    if not rows:
        print("Project is ready. No data files are configured yet.")
        print("Add data to data/raw/, pass --source PATH, or copy configs/data_sources.example.toml")
        print("to configs/data_sources.local.toml and set source_path. No analysis has been run.")
        return
    summary = summarize(rows)
    write_inventory(rows, REPORTS_DIR / "data_inventory.csv")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    table = None
    if config.get("table_path"):
        table = read_table(resolve_project_path(config["table_path"]))
    elif config.get("target_member"):
        table = read_table(source, member=config["target_member"])
    elif source.is_file() and source.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xlsm"}:
        table = read_table(source)
    if table is not None:
        print("\nTable preview (original columns, no transformations):")
        print(table.head().to_string(index=False))
        print(f"Rows: {len(table)}; columns: {list(table.columns)}")
    print("\nInventory saved in reports/. Add your analysis to a script in scripts/.")


# ----------------------------------------------------------------------------
# hash: exact duplicates among the raw files
# ----------------------------------------------------------------------------


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def hash_raw_files() -> None:
    """SHA-256 of every file under data/raw and groups of exact duplicates."""
    TABLES.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW.rglob("*") if p.is_file() and p.name != ".gitkeep")
    rows = []
    for index, path in enumerate(files, 1):
        rel = path.relative_to(RAW).as_posix()
        rows.append({"path": rel, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
        if index % 50 == 0:
            print(f"{index}/{len(files)} hashed", flush=True)

    with (TABLES / "raw_file_hashes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[row["sha256"]].append(row)
    duplicates = [g for g in groups.values() if len(g) > 1]
    with (TABLES / "raw_exact_duplicates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "sha256", "size_bytes", "path"])
        writer.writeheader()
        for number, group in enumerate(sorted(duplicates, key=lambda g: g[0]["path"]), 1):
            for row in group:
                writer.writerow({"group": number, **row})
    wasted = sum(g[0]["size_bytes"] * (len(g) - 1) for g in duplicates)
    print(f"{len(rows)} files, {len(duplicates)} duplicate groups, {wasted / 1e9:.3f} GB in extra copies")


# ----------------------------------------------------------------------------
# check-grids: dates and copied months in the wind and precipitation grids
# ----------------------------------------------------------------------------


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()[:16]


def wind_rows():
    for path in sorted(WIND_DIR.glob("*/CCMP_Wind_Analysis_*_L4.5.nc")):
        nominal = re.search(r"_(\d{6})_", path.name).group(1)
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            base = re.match(r"Y(\d{4}) M(\d{2})", ds.getncattr("base_date"))
            stored = base.group(1) + base.group(2)
            w = np.squeeze(ds["w"][:]).astype("float64")
            u = np.squeeze(ds["u"][:])
            fill = getattr(ds["w"], "_FillValue", None)
            bad = ~np.isfinite(w) | ((w == fill) if fill is not None else False)
            good = w[~bad]
            yield {
                "nominal_month": f"{nominal[:4]}-{nominal[4:]}",
                "stored_month": f"{stored[:4]}-{stored[4:]}",
                "date_matches": nominal == stored,
                "version": ds.getncattr("product_version"),
                "w_hash": array_hash(np.squeeze(ds["w"][:])),
                "u_hash": array_hash(u),
                "invalid_fraction": round(float(bad.mean()), 4),
                "w_min": round(float(good.min()), 3),
                "w_max": round(float(good.max()), 3),
                "w_mean": round(float(good.mean()), 4),
                "path": path.relative_to(RAW).as_posix(),
            }


def precip_rows():
    for path in sorted(PRECIP_DIR.glob("ERA5_Total_Precipitation_*.nc")):
        year = re.search(r"_(\d{4})\.nc$", path.name).group(1)
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            times = netCDF4.num2date(ds["valid_time"][:], ds["valid_time"].units)
            tp = ds["tp"]
            for index, when in enumerate(times):
                field = np.asarray(tp[index], dtype="float64")
                bad = ~np.isfinite(field)
                good = field[~bad]
                yield {
                    "file_year": year,
                    "stored_month": f"{when.year:04d}-{when.month:02d}",
                    "year_matches": str(when.year) == year,
                    "tp_hash": array_hash(np.asarray(tp[index])),
                    "invalid_fraction": round(float(bad.mean()), 4),
                    "tp_min_m_per_day": float(good.min()),
                    "tp_max_m_per_day": float(good.max()),
                    "tp_mean_m_per_day": float(good.mean()),
                    "path": path.relative_to(RAW).as_posix(),
                }


def flag_repeats(frame: pd.DataFrame, column: str, month_column: str) -> pd.DataFrame:
    groups = defaultdict(list)
    for month, value in zip(frame[month_column], frame[column]):
        groups[value].append(month)
    frame["same_data_as"] = [
        ";".join(m for m in groups[value] if m != month) for month, value in zip(frame[month_column], frame[column])
    ]
    return frame


def check_grids() -> None:
    """Stored date versus file name, data hash, fill share and range of every wind and precipitation grid.

    Writes reports/tables/wind_month_checks.csv and precip_month_checks.csv."""
    TABLES.mkdir(parents=True, exist_ok=True)
    wind = flag_repeats(pd.DataFrame(wind_rows()), "w_hash", "nominal_month")
    wind.to_csv(TABLES / "wind_month_checks.csv", index=False)
    precip = flag_repeats(pd.DataFrame(precip_rows()), "tp_hash", "stored_month")
    precip.to_csv(TABLES / "precip_month_checks.csv", index=False)

    for name, frame, month in (("wind", wind, "nominal_month"), ("precipitation", precip, "stored_month")):
        print(f"\n== {name}: {len(frame)} months")
        date_col = "date_matches" if name == "wind" else "year_matches"
        print("date/filename mismatches:", frame.loc[~frame[date_col], month].tolist())
        print("months sharing data with another month:", frame.loc[frame["same_data_as"] != "", month].tolist())
        months = pd.PeriodIndex(frame[month], freq="M")
        full = pd.period_range(months.min(), months.max(), freq="M")
        print("range:", months.min(), "to", months.max(), "| missing:", [str(m) for m in full.difference(months)])
        print("duplicate months:", frame.loc[frame[month].duplicated(), month].tolist())
        print("max invalid fraction:", frame["invalid_fraction"].max())
    print(wind.groupby("version")[["w_mean", "invalid_fraction"]].describe().T.round(3))
    print(precip["tp_mean_m_per_day"].describe())


# ----------------------------------------------------------------------------
# oras5-download and oras5-process: ORAS5 depth of the 20 °C isotherm
# ----------------------------------------------------------------------------

ORAS5_DATASET = "reanalysis-oras5"
ORAS5_VARIABLE = "depth_of_20_c_isotherm"


def request_year(client: cdsapi.Client, year: int, months: list[int] | None = None) -> Path:
    target = ORAS5_ARCHIVES / f"oras5_d20_{year}.zip"
    if target.exists() and target.stat().st_size > 0:
        print(f"{year}: already downloaded ({target.stat().st_size / 1e6:.1f} MB)")
        return target
    request = {
        "product_type": ["consolidated" if year <= 2014 else "operational"],
        "vertical_resolution": "single_level",
        "variable": [ORAS5_VARIABLE],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in (months or range(1, 13))],
    }
    partial = target.with_suffix(".part")
    client.retrieve(ORAS5_DATASET, request).download(str(partial))
    partial.rename(target)
    print(f"{year}: {target.stat().st_size / 1e6:.1f} MB")
    return target


def download_oras5() -> None:
    """Download ORAS5 monthly D20 from the Copernicus Climate Data Store: FIRST_YEAR [LAST_YEAR] [--process].

    With --process, each year is reduced by process_archive and its archive deleted before the next
    year is downloaded, so at most one archive is on disk. Needs a CDS account, the ORAS5 licence
    accepted on the dataset page, and the API key in ~/.cdsapirc (created by the user; this command
    never reads or prints it). Archives go to data/external/oras5/ (not in Git), one per year; years
    already downloaded are skipped. ORAS5 "consolidated" covers 1958-2014 and "operational" covers
    2015 onwards."""
    if not (Path.home() / ".cdsapirc").exists():
        sys.exit("No ~/.cdsapirc found. Create it with your CDS url and key first.")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    process = "--process" in sys.argv
    first = int(args[0])
    last = int(args[1]) if len(args) > 1 else first
    ORAS5_ARCHIVES.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(quiet=True, progress=False)
    for year in range(first, last + 1):
        if process and (ORAS5_DIR / f"d20_indices_{year}.csv").exists():
            print(f"{year}: already processed", flush=True)
            continue
        months = None
        if year == date.today().year:  # current year: only months already published
            months = list(range(1, date.today().month))
        try:
            archive = request_year(client, year, months)
        except Exception as error:  # noqa: BLE001
            print(f"{year}: download failed: {str(error)[:200]}", flush=True)
            continue
        if process:
            process_archive(archive, delete=True)
            print(f"{year}: processed and archive deleted", flush=True)


# (lat_min, lat_max, lon_min, lon_max) in degrees east 0-360
ORAS5_BOXES = {
    "d20_wwv": (-5, 5, 120, 280),        # equatorial Pacific warm water volume region
    "d20_wwv_west": (-5, 5, 120, 205),   # 120E-155W (Meinen & McPhaden 2000)
    "d20_wwv_east": (-5, 5, 205, 280),   # 155W-80W
    "d20_nino34": (-5, 5, 190, 240),
    "d20_eqio": (-5, 5, 50, 100),
}
BAND_LAT = np.arange(-19.5, 20.0, 1.0)
BAND_LON = np.arange(0.5, 360.0, 1.0)


def read_month(path: Path):
    with netCDF4.Dataset(path) as ds:
        lat = np.asarray(ds["nav_lat"][:], dtype="float64")
        lon = np.mod(np.asarray(ds["nav_lon"][:], dtype="float64"), 360)
        d20 = np.ma.filled(ds["so20chgt"][0].astype("float64"), np.nan)
        times = netCDF4.num2date(ds["time_counter"][:], ds["time_counter"].units)
    when = pd.Timestamp(times[0].year, times[0].month, 1)
    d20[d20 <= 0] = np.nan  # land / no 20 C water
    return when, lat, lon, d20


def box_means(lat, lon, d20) -> dict[str, float]:
    weight = np.cos(np.deg2rad(lat)) ** 2
    out = {}
    for name, (la0, la1, lo0, lo1) in ORAS5_BOXES.items():
        mask = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1) & np.isfinite(d20)
        out[name] = float(np.sum(d20[mask] * weight[mask]) / np.sum(weight[mask]))
        out[name + "_cells"] = int(mask.sum())
    return out


def band_field(lat, lon, d20) -> np.ndarray:
    keep = (np.abs(lat) < 20) & np.isfinite(d20)
    i = np.floor(lat[keep] + 20).astype(int)
    j = np.floor(lon[keep]).astype(int) % 360
    total = np.zeros((40, 360)); count = np.zeros((40, 360))
    np.add.at(total, (i, j), d20[keep]); np.add.at(count, (i, j), 1)
    with np.errstate(invalid="ignore"):
        return (total / count).astype("float32")


def process_archive(archive: Path, delete: bool = False) -> Path:
    year = re.search(r"(\d{4})", archive.stem).group(1)
    ORAS5_DIR.mkdir(parents=True, exist_ok=True)
    rows, fields, months = [], [], []
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as tmp:
        names = sorted(n for n in z.namelist() if n.endswith(".nc") and "/" not in n and ".." not in n)
        for name in names:
            path = Path(z.extract(name, tmp))
            when, lat, lon, d20 = read_month(path)
            rows.append({"date": when, **box_means(lat, lon, d20)})
            fields.append(band_field(lat, lon, d20)); months.append(str(when.date()))
            path.unlink()
    if len(rows) != 12 and int(year) < pd.Timestamp.now().year:
        raise ValueError(f"{archive.name}: expected 12 months, found {len(rows)}")
    indices_path = ORAS5_DIR / f"d20_indices_{year}.csv"
    pd.DataFrame(rows).sort_values("date").to_csv(indices_path, index=False)
    np.savez_compressed(ORAS5_DIR / f"d20_band_{year}.npz", d20=np.stack(fields), months=np.array(months),
                        lat=BAND_LAT, lon=BAND_LON)
    if delete:
        archive.unlink()
    return indices_path


def process_oras5() -> None:
    """Reduce the ORAS5 archives in data/external/oras5/ to what the study uses [--delete].

    For each yearly archive this writes monthly box means of D20 (m) to
    data/interim/oras5/d20_indices_<year>.csv, for the warm-water-volume region and its western and
    eastern parts, the Niño-3.4 box and the equatorial Indian Ocean, and a 1-degree, 20S-20N binned
    D20 field to d20_band_<year>.npz. With --delete, the archive is removed once both are written.

    Box means weight cells by cos(lat)^2, the approximate area of an isotropic ORCA025 cell; within
    5S-5N the weights differ by under 1%. Warm water volume is the box area times the mean D20
    (Meinen & McPhaden 2000), so the mean D20 anomaly is used as the WWV index."""
    delete = "--delete" in sys.argv
    for archive in sorted(ORAS5_ARCHIVES.glob("oras5_d20_*.zip")):
        print("processed", process_archive(archive, delete).name)


# ----------------------------------------------------------------------------
# build: the monthly and quarterly ENSO dataset
# ----------------------------------------------------------------------------

SST_DIR = source_directory("sst", "sst", "SST Data")
ERSST_V6 = SST_DIR / "ersst.v6.1850-last.nc"
if not ERSST_V6.is_file() and (SST_DIR / "SST (DMI)" / ERSST_V6.name).is_file():
    ERSST_V6 = SST_DIR / "SST (DMI)" / ERSST_V6.name
ERSST_V5 = SST_DIR / "sst.mnmean.nc"
WIND_DIR = source_directory("wind", "CCMP_V03.0_Monthly")
PRECIP_DIR = source_directory("precipitation", "ERA5_Precipitation_1975_2026")
OHC_DIR = source_directory("ohc", "ohc", "Ocean Heat Content")

WIND_EXCLUDED = {f"2019-{m:02d}" for m in range(1, 10)}
WIND_REPLACEMENT_DIR = ROOT / "data" / "external" / "ccmp_v3.0_2019_replacement"

# (lat_min, lat_max, lon_min, lon_max), degrees east 0-360
WIND_BOXES = {
    "u_cpac": (-5, 5, 170, 220),  # central Pacific trade winds (supplied MATLAB)
    "u_epac": (-5, 5, 220, 280),  # eastern Pacific trade winds (supplied MATLAB)
    "u_wpac": (-5, 5, 120, 170),  # western Pacific, westerly wind bursts
    "u_eqio": (-5, 5, 70, 90),    # equatorial Indian Ocean, IOD-related
}
PRECIP_BOXES = {
    "pr_nino4": (-5, 5, 160, 210),
    "pr_nino34": (-5, 5, 190, 240),
    "pr_wtio": (-10, 10, 50, 70),
    "pr_etio": (-10, 0, 90, 110),
}
OHC_BOXES = {
    "ohc700_nino34": (-5, 5, 190, 240),
    "ohc700_eqpac": (-5, 5, 120, 280),  # warm-water-volume style region
}


def box_index(lat: np.ndarray, lon: np.ndarray, box) -> tuple[np.ndarray, np.ndarray]:
    lat_min, lat_max, lon_min, lon_max = box
    lon360 = np.mod(lon, 360)
    return (lat >= lat_min) & (lat <= lat_max), (lon360 >= lon_min) & (lon360 <= lon_max)


def weighted_mean(field: np.ndarray, lat: np.ndarray) -> float:
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], field.shape)
    valid = np.isfinite(field)
    return float(np.sum(field[valid] * weights[valid]) / np.sum(weights[valid]))


def sst_indices() -> pd.DataFrame:
    v6 = load_ersst(ERSST_V6)
    # Raw (absolute) SST is kept so forecast evaluation can compute anomalies from
    # training years only; the *_ssta columns use the fixed 1991-2020 climatology.
    frame = pd.DataFrame({f"{r}_sst": box_mean(v6, r, weighted=True) for r in ("nino34", "wtio", "etio", "tropics")})
    for region in ("nino34", "wtio", "etio"):
        frame[f"{region}_ssta"] = monthly_anomaly(frame[f"{region}_sst"])
    frame["dmi"] = frame["wtio_ssta"] - frame["etio_ssta"]
    tropics = monthly_anomaly(frame["tropics_sst"])
    frame["tropics_ssta"] = tropics
    frame["nino34_rel_ssta"] = relative_anomaly(frame["nino34_ssta"], tropics)
    v5_grid = load_ersst(ERSST_V5)
    v5 = box_mean(v5_grid, "nino34", weighted=False)  # reproduces the supplied workbook
    frame["nino34_ssta_v5"] = monthly_anomaly(v5).reindex(frame.index)
    # ERSST v5 with the same weighted boxes as v6, for the v5 sensitivity test.
    for region in ("nino34", "tropics"):
        frame[f"{region}_sst_v5"] = box_mean(v5_grid, region, weighted=True).reindex(frame.index)
    return frame.loc["1950":]


def wind_indices() -> pd.DataFrame:
    records = []
    for path in sorted(WIND_DIR.glob("*/CCMP_Wind_Analysis_*_L4.5.nc")):
        stamp = re.search(r"_(\d{4})(\d{2})_", path.name)
        month = f"{stamp.group(1)}-{stamp.group(2)}"
        row = {"date": pd.Timestamp(f"{month}-01"), "wind_source": "supplied"}
        if month in WIND_EXCLUDED:
            path = WIND_REPLACEMENT_DIR / path.name
            if not path.is_file():
                row["wind_source"] = "missing"
                records.append(row)
                continue
            row["wind_source"] = "rss_download"
        with netCDF4.Dataset(path) as ds:
            stored = re.match(r"Y(\d{4}) M(\d{2})", ds.getncattr("base_date"))
            if f"{stored.group(1)}-{stored.group(2)}" != month:
                raise ValueError(f"{path}: base_date {ds.base_date} does not match {month}")
            lat = np.asarray(ds["latitude"][:], dtype="float64")
            lon = np.asarray(ds["longitude"][:], dtype="float64")
            for name, box in WIND_BOXES.items():
                lat_ok, lon_ok = box_index(lat, lon, box)
                u = np.squeeze(ds["u"][:]).filled(np.nan).astype("float64")
                row[name] = weighted_mean(u[np.ix_(lat_ok, lon_ok)], lat[lat_ok])
        records.append(row)
    frame = pd.DataFrame(records).set_index("date")
    for column in WIND_BOXES:
        frame[f"{column}_anom"] = monthly_anomaly(frame[column], 1993, 2020)
    return frame


def precip_indices() -> pd.DataFrame:
    records = []
    for path in sorted(PRECIP_DIR.glob("ERA5_Total_Precipitation_*.nc")):
        with netCDF4.Dataset(path) as ds:
            lat = np.asarray(ds["latitude"][:], dtype="float64")
            lon = np.asarray(ds["longitude"][:], dtype="float64")
            times = netCDF4.num2date(ds["valid_time"][:], ds["valid_time"].units)
            boxes = {name: box_index(lat, lon, box) for name, box in PRECIP_BOXES.items()}
            for index, when in enumerate(times):
                field = np.asarray(ds["tp"][index], dtype="float64") * 1000.0  # m/day -> mm/day
                row = {"date": pd.Timestamp(when.year, when.month, 1)}
                for name, (lat_ok, lon_ok) in boxes.items():
                    row[name] = weighted_mean(field[np.ix_(lat_ok, lon_ok)], lat[lat_ok])
                records.append(row)
    frame = pd.DataFrame(records).set_index("date").sort_index()
    for column in PRECIP_BOXES:
        frame[f"{column}_anom"] = monthly_anomaly(frame[column])
    return frame


def ohc_quarterly() -> pd.DataFrame:
    columns = {}
    sources = {"ohc700": "heat_content_anomaly_0-700_seasonal.nc", "ohc2000": "heat_content_anomaly_0-2000_seasonal.nc"}
    for prefix, filename in sources.items():
        with netCDF4.Dataset(OHC_DIR / filename) as ds:
            bounds = np.asarray(ds["climatology_bounds"][:], dtype="float64")
            if not np.allclose(bounds[:, 1] - bounds[:, 0], 3) or not np.allclose(bounds % 3, 0):
                raise ValueError(f"{filename}: intervals are not calendar quarters")
            starts = pd.DatetimeIndex([pd.Timestamp("1955-01-01") + pd.DateOffset(months=int(b)) for b in bounds[:, 0]])
            quarters = starts.to_period("Q")
            lat = np.asarray(ds["lat"][:], dtype="float64")
            lon = np.asarray(ds["lon"][:], dtype="float64")
            heat = np.squeeze(ds["h18_hc"][:]).filled(np.nan)
            for region, box in OHC_BOXES.items():
                name = region.replace("ohc700", prefix)
                lat_ok, lon_ok = box_index(lat, lon, box)
                sub = heat[:, lat_ok][:, :, lon_ok]
                cells = np.isfinite(sub).sum(axis=(1, 2))
                if cells.min() != cells.max():
                    raise ValueError(f"{name}: ocean-cell count changes over time")
                columns[name] = pd.Series(np.nansum(sub, axis=(1, 2)), index=quarters)
    frame = pd.DataFrame(columns)
    frame.index.name = "quarter"
    return frame


def oras5_indices() -> pd.DataFrame:
    """Monthly ORAS5 20 C isotherm depth (m) box means; deeper = more heat stored.
    d20_wwv is the warm-water-volume index (5S-5N, 120E-80W). Empty if not downloaded."""
    files = sorted(ORAS5_DIR.glob("d20_indices_*.csv"))
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files]).set_index("date").sort_index()
    if frame.index.duplicated().any():
        raise ValueError("duplicate ORAS5 months")
    return frame[[c for c in frame.columns if not c.endswith("_cells")]]


def oni_and_phase(oni: pd.Series) -> pd.DataFrame:
    """NOAA-style phase from a 3-month index: >=5 overlapping seasons beyond +/-0.5 C."""
    phase = pd.Series("neutral", index=oni.index)
    for sign, label in ((1, "el_nino"), (-1, "la_nina")):
        beyond = (sign * oni >= 0.5).astype(int)
        run_id = (beyond.diff() != 0).cumsum()
        run_length = beyond.groupby(run_id).transform("sum")
        phase[(beyond == 1) & (run_length >= 5)] = label
    phase[oni.isna()] = pd.NA
    return pd.DataFrame({"oni": oni, "enso_phase": phase})


def build_dataset() -> None:
    """Build the unified ENSO predictor dataset from the verified raw sources.

    Outputs (data/processed/):
      enso_monthly.csv    one row per calendar month, 1950-01 to 2026-08
      enso_quarterly.csv  calendar quarters (Jan-Mar ... Oct-Dec), for ocean heat content

    Data preparation choices:
      * SST: ERSST v6, one canonical copy, cos(latitude)-weighted box means.
        Niño-3.4 from ERSST v5 (the version behind the supplied workbook) is kept
        as a comparison column.
      * ENSO phase from both the fixed-baseline ONI and RONI (NOAA CPC's official
        index since Feb 2026: 3-month Niño-3.4 minus 20S-20N mean anomaly, rescaled
        per calendar month over 1950-2020; L'Heureux et al. 2024).
      * Anomalies: calendar-month climatology 1991-2020, except CCMP wind which
        starts in 1993 and uses 1993-2020 (as in the supplied MATLAB scripts).
      * CCMP 2019-01..2019-09 in the supplied bundle are copies of 2016. They are
        replaced by files downloaded from RSS on 2026-10-18 into
        data/external/ccmp_v3.0_2019_replacement/ (left missing if absent). Every
        wind file's base_date must match its filename month.
      * Precipitation: ERA5 monthly mean of daily totals, converted to mm/day.
      * Ocean heat content: NOAA/NCEI 1-degree quarterly anomalies, summed over the
        region (10^18 J), kept quarterly; no monthly values are invented.
    Raw files are only read."""

    PROCESSED.mkdir(parents=True, exist_ok=True)
    sst = sst_indices()
    print("SST indices:", sst.index.min().date(), "to", sst.index.max().date())
    wind = wind_indices()
    print("wind indices:", wind.index.min().date(), "to", wind.index.max().date())
    precip = precip_indices()
    print("precipitation indices:", precip.index.min().date(), "to", precip.index.max().date())

    oni = sst["nino34_ssta"].rolling(3, center=True).mean()
    roni = relative_anomaly(oni, sst["tropics_ssta"].rolling(3, center=True).mean())
    relative = oni_and_phase(roni).rename(columns={"oni": "roni", "enso_phase": "enso_phase_relative"})
    monthly = sst.join(oni_and_phase(oni)).join(relative).join(wind, how="left").join(precip, how="left")
    monthly = monthly.join(oras5_indices(), how="left")
    monthly.index.name = "date"
    monthly.round(5).to_csv(PROCESSED / "enso_monthly.csv")

    quarterly = ohc_quarterly()
    means = monthly.drop(columns=["enso_phase", "enso_phase_relative", "wind_source"]).groupby(monthly.index.to_period("Q")).agg(["mean", "count"])
    complete = means.xs("count", axis=1, level=1) == 3
    quarter_means = means.xs("mean", axis=1, level=1).where(complete)
    quarterly = quarterly.join(quarter_means.add_suffix("_qmean"), how="outer")
    quarterly.index = quarterly.index.astype(str)
    quarterly.round(5).to_csv(PROCESSED / "enso_quarterly.csv")
    print(f"monthly: {monthly.shape}, quarterly: {quarterly.shape}")
    print(monthly.notna().sum().to_string())


# ----------------------------------------------------------------------------
# grids: SST and D20 maps for the gridded networks (E8)
# ----------------------------------------------------------------------------

GRIDS = PROCESSED / "tropical_grids.npz"
LAT_EDGE = 18.0
GRID_START = "1958-01-01"


def oras5_on_ersst_grid(lat: np.ndarray, lon: np.ndarray) -> pd.Series:
    """D20 per month on the ERSST cells, as a Series of 2-D arrays indexed by month."""
    fields, months = [], []
    for path in sorted(ORAS5_DIR.glob("d20_band_*.npz")):
        band = np.load(path)
        d20, blat, blon = band["d20"], band["lat"], band["lon"]
        rows = [np.flatnonzero(np.isin(blat, (y - 0.5, y + 0.5))) for y in lat]
        cols = [np.flatnonzero(np.isin(blon, ((x - 0.5) % 360, (x + 0.5) % 360))) for x in lon]
        if any(len(r) != 2 for r in rows) or any(len(c) != 2 for c in cols):
            raise ValueError(f"{path.name}: ORAS5 cells do not tile the ERSST grid")
        r, c = np.array(rows), np.array(cols)
        # (month, lat, 2, lon, 2) -> mean over the 2 x 2 sub-cells, ignoring land
        block = d20[:, r[:, :, None, None], c[None, None, :, :]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-land cells give empty means
            fields.append(np.nanmean(block.reshape(len(d20), len(lat), 2, len(lon), 2), axis=(2, 4)))
        months.extend(pd.to_datetime(band["months"]))
    return pd.Series(list(np.concatenate(fields)), index=pd.DatetimeIndex(months))


def build_grids() -> None:
    """Monthly tropical grids for the gridded deep-learning baseline (E8): data/processed/tropical_grids.npz.

    SST from ERSST v6 and the 20 °C isotherm depth (D20) from ORAS5 on one 2° grid, 18S-18N, all
    longitudes, 1958 onwards. ERSST cells are centred on even degrees; each is matched by the mean of
    the 2 x 2 ORAS5 1° cells it contains (the band files of oras5-process). Values are absolute;
    anomalies are computed inside each training window."""
    sst = load_ersst(ERSST_V6, lat_range=(-LAT_EDGE, LAT_EDGE)).sel(time=slice(GRID_START, None))
    lat, lon = sst.lat.to_numpy(), sst.lon.to_numpy()
    d20 = oras5_on_ersst_grid(lat, lon)
    months = sst.indexes["time"].intersection(d20.index)
    sst_grid = sst.sel(time=months).to_numpy().astype("float32")
    d20_grid = np.stack(d20.loc[months].to_list()).astype("float32")
    ocean = np.isfinite(sst_grid).all(axis=0) & np.isfinite(d20_grid).all(axis=0)
    np.savez_compressed(GRIDS, months=months.strftime("%Y-%m-%d").to_numpy().astype("U10"), lat=lat, lon=lon,
                        sst=sst_grid, d20=d20_grid, ocean=ocean)
    print(f"{GRIDS.name}: {len(months)} months {months[0]:%Y-%m} to {months[-1]:%Y-%m}, grid {len(lat)} x {len(lon)}, "
          f"ocean cells {ocean.sum()} ({ocean.mean():.0%})")


# ----------------------------------------------------------------------------
# long-indices: SST indices from 1870 (E9)
# ----------------------------------------------------------------------------

LONG_START = "1870-01-01"


def build_long_indices() -> None:
    """Long monthly SST indices (ERSST v6, 1870 onwards) for E9: data/processed/enso_monthly_long.csv.

    Same cos(latitude)-weighted boxes as build, so values from 1950 match enso_monthly.csv. WWV
    columns are empty before 1958."""
    v6 = load_ersst(ERSST_V6).sel(time=slice(LONG_START, None))
    frame = pd.DataFrame({r: box_mean(v6, r, weighted=True) for r in ("nino34", "wtio", "etio", "tropics")})
    monthly = load_monthly()
    check = (frame.loc["1950":, "nino34"] - monthly["nino34_sst"]).abs().max()
    if check > 1e-4:
        raise ValueError(f"long Niño-3.4 differs from enso_monthly.csv by {check}")
    frame = frame.join(monthly[["d20_wwv", "d20_wwv_west", "d20_wwv_east"]])
    frame.index.name = "date"
    frame.to_csv(PROCESSED / "enso_monthly_long.csv")
    print(f"{len(frame)} months {frame.index[0]:%Y-%m} to {frame.index[-1]:%Y-%m}; "
          f"WWV from {frame.d20_wwv.first_valid_index():%Y-%m}; max diff vs dataset {check:.2e}")


COMMANDS = {
    "inventory": inventory_report,
    "hash": hash_raw_files,
    "check-grids": check_grids,
    "oras5-download": download_oras5,
    "oras5-process": process_oras5,
    "build": build_dataset,
    "grids": build_grids,
    "long-indices": build_long_indices,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        names = "\n".join(f"  {name:<16} {function.__doc__.splitlines()[0]}" for name, function in COMMANDS.items())
        sys.exit(f"usage: python {Path(sys.argv[0]).name} COMMAND [options]\n\ncommands:\n{names}")
    command = sys.argv.pop(1)  # the remaining arguments are read by the command, as in the original scripts
    if command != "inventory":
        ensure_output_directories()
    COMMANDS[command]()


if __name__ == "__main__":
    main()
