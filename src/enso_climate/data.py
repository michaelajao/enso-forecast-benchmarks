"""Raw data: file inventories, file inspection and table reading, and regional SST indices.

Inventories list archive contents or local files without opening their data arrays. ZIP members
are copied to a temporary file using a fixed safe filename, never their embedded paths, and
metadata inspection does not read complete NetCDF grids.

SST index times are mapped to the first day of each calendar month. ERSST v6 stores
"months since 1850-01-01", which the default decoders reject, so integer month offsets are
converted explicitly. This module needs netCDF4 and xarray (the project environment).
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import csv
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from .paths import DATA_DIR


# ---------------------------------------------------------------------------
# Inventories
# ---------------------------------------------------------------------------

FIELDS = ["path", "group", "extension", "size_bytes", "compressed_bytes"]


def inventory(source: str | Path) -> list[dict]:
    """One row (path, group, extension, sizes) per file in a directory, single file or ZIP archive."""
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Data source does not exist: {source}")
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            return [
                {
                    "path": item.filename,
                    "group": PurePosixPath(item.filename).parts[0],
                    "extension": PurePosixPath(item.filename).suffix.lower(),
                    "size_bytes": item.file_size,
                    "compressed_bytes": item.compress_size,
                }
                for item in archive.infolist()
                if not item.is_dir()
            ]
    paths = sorted(source.rglob("*")) if source.is_dir() else [source]
    result = []
    for path in paths:
        if not path.is_file() or path.name == ".gitkeep" or any(p in {".git", ".venv", "__pycache__"} for p in path.parts):
            continue
        name = path.relative_to(source).as_posix() if source.is_dir() else path.name
        result.append({
            "path": name,
            "group": PurePosixPath(name).parts[0],
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "compressed_bytes": None,
        })
    return result


def summarize(rows: list[dict]) -> dict:
    """File count, total bytes and counts by extension and top-level group."""
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "extensions": dict(sorted(Counter(row["extension"] for row in rows).items())),
        "groups": dict(sorted(Counter(row["group"] for row in rows).items())),
    }


def write_inventory(rows: list[dict], destination: str | Path) -> Path:
    """Write inventory rows to a CSV file, creating its folder."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return destination


# ---------------------------------------------------------------------------
# Inspecting files and reading tables
# ---------------------------------------------------------------------------

MAX_ARCHIVE_MEMBER_BYTES = 256 * 1024 * 1024


@contextmanager
def local_file(source: str | Path, member: str | None = None):
    """Yield a local path for a file, or for one ZIP member copied to a temporary file."""
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"File not found: {source}")
    if member is None:
        if source.suffix.lower() == ".zip":
            raise ValueError("Choose a file inside the archive with --member; use inventory to list names.")
        yield source
        return
    if source.suffix.lower() != ".zip":
        raise ValueError("A member can only be selected from a ZIP archive.")
    with zipfile.ZipFile(source) as archive:
        info = archive.getinfo(member)
        if info.is_dir():
            raise ValueError("Choose a file, not an archive directory.")
        if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise ValueError(
                "This file exceeds the 256 MiB archive inspection limit. "
                "Extract that file separately and inspect its local path."
            )
        scratch = DATA_DIR / "interim"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="enso-inspect-", dir=scratch) as temporary:
            destination = Path(temporary) / ("sample" + Path(member).suffix.lower())
            with archive.open(info) as reader, destination.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
            yield destination


def read_table(source: str | Path, member: str | None = None, **kwargs):
    """Read a CSV, TSV or Excel table. No dates, anomalies or units are assumed."""
    import pandas as pd

    with local_file(source, member) as path:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, **kwargs)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", **kwargs)
        if suffix in {".xlsx", ".xlsm"}:
            return pd.read_excel(path, **kwargs)
        raise ValueError(f"Unsupported table format: {suffix}. Use CSV, TSV or XLSX.")


def inspect_file(source: str | Path, member: str | None = None) -> dict:
    """Return shapes, fields and small previews; never execute source scripts."""
    with local_file(source, member) as path:
        result = {
            "source": str(Path(source).expanduser().resolve()),
            "member": member,
            "size_bytes": path.stat().st_size,
            "extension": path.suffix.lower(),
        }
        suffix = path.suffix.lower()
        if suffix in {".nc", ".nc4"}:
            from netCDF4 import Dataset, num2date

            with Dataset(path) as dataset:
                result["attributes"] = {
                    name: str(dataset.getncattr(name))
                    for name in ("title", "institution", "source", "version", "product_version", "base_date")
                    if name in dataset.ncattrs()
                }
                result["dimensions"] = {key: len(value) for key, value in dataset.dimensions.items()}
                result["variables"] = {}
                for name, variable in dataset.variables.items():
                    details = {
                        "dimensions": list(variable.dimensions),
                        "shape": list(variable.shape),
                        "dtype": str(variable.dtype),
                    }
                    for attr in ("units", "long_name", "standard_name", "calendar"):
                        if attr in variable.ncattrs():
                            details[attr] = str(variable.getncattr(attr))
                    if name.lower() in {"time", "valid_time"} and variable.ndim == 1 and variable.size:
                        details["first_value"] = str(variable[0])
                        details["last_value"] = str(variable[-1])
                        units = getattr(variable, "units", "")
                        if "since" in units:
                            calendar = getattr(variable, "calendar", "standard")
                            try:
                                details["first_date"] = str(num2date(variable[0], units, calendar))
                                details["last_date"] = str(num2date(variable[-1], units, calendar))
                            except (ValueError, TypeError, OverflowError) as error:
                                details["time_decode_note"] = (
                                    f"Automatic date decoding unavailable: {error}. "
                                    "Raw endpoints and units are retained. Verify the product's "
                                    "calendar and interval bounds before interpreting dates."
                                )
                    result["variables"][name] = details
        elif suffix == ".mat":
            import h5py
            from scipy.io import whosmat

            if h5py.is_hdf5(path):
                with h5py.File(path, "r") as dataset:
                    variables = {}
                    def visit(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            variables[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
                    dataset.visititems(visit)
                    result["variables"] = variables
                    result["format"] = "HDF5 / MATLAB v7.3"
            else:
                result["format"] = "MATLAB v4-v7.2"
                try:
                    result["variables"] = {
                        name: {"shape": list(shape) if shape is not None else None, "dtype": dtype}
                        for name, shape, dtype in whosmat(path)
                    }
                except (TypeError, NotImplementedError) as error:
                    result["variables"] = None
                    result["metadata_read_note"] = (
                        f"SciPy could not enumerate variables: {error}. This can occur "
                        "with MATLAB object metadata. Inspect in MATLAB or export selected "
                        "numeric variables separately; the variable list remains unknown."
                    )
        elif suffix in {".xlsx", ".xlsm"}:
            from openpyxl import load_workbook

            workbook = load_workbook(path, read_only=True, data_only=True)
            try:
                result["sheets"] = {
                    sheet.title: {
                        "rows": sheet.max_row,
                        "columns": sheet.max_column,
                        "preview": [list(row) for row in sheet.iter_rows(max_row=6, values_only=True)],
                    }
                    for sheet in workbook
                }
            finally:
                workbook.close()
        elif suffix in {".csv", ".tsv"}:
            frame = read_table(path, nrows=5)
            result["columns"] = list(frame.columns)
            result["preview"] = frame.to_dict(orient="records")
        elif suffix in {".txt", ".m", ".py"}:
            with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
                result["preview"] = [line.rstrip() for _, line in zip(range(20), stream)]
        else:
            result["note"] = "No format-specific reader; inventory metadata only."
        return result


# ---------------------------------------------------------------------------
# Regional SST indices from ERSST monthly grids
# ---------------------------------------------------------------------------

# (lat_min, lat_max, lon_min, lon_max), longitudes in degrees east 0-360.
REGIONS = {
    "nino34": (-5.0, 5.0, 190.0, 240.0),
    "wtio": (-10.0, 10.0, 50.0, 70.0),
    "etio": (-10.0, 0.0, 90.0, 110.0),
    "tropics": (-20.0, 20.0, 0.0, 360.0),
    "atl3": (-3.0, 3.0, 340.0, 360.0),  # Atlantic Niño (E14)
    "nta": (5.0, 25.0, 305.0, 345.0),  # north tropical Atlantic (E14)
}


def _month_starts(variable) -> pd.DatetimeIndex:
    units = variable.units
    values = np.asarray(variable[:], dtype="float64")
    if units.startswith("months since"):
        if not np.allclose(values, np.round(values)):
            raise ValueError("Non-integer month offsets; map interval bounds explicitly.")
        origin = pd.Timestamp(units.split("since", 1)[1].strip()[:10])
        return pd.DatetimeIndex([origin + pd.DateOffset(months=int(v)) for v in values])
    dates = netCDF4.num2date(values, units, getattr(variable, "calendar", "standard"))
    return pd.DatetimeIndex([pd.Timestamp(d.year, d.month, 1) for d in dates])


def load_ersst(path: str | Path, lat_range: tuple[float, float] = (-25.0, 25.0)) -> xr.DataArray:
    """Return SST (time, lat, lon) with ascending latitude and month-start times.

    Only latitudes within ``lat_range`` are read (every index in REGIONS lies in
    20S-20N), as float32, to keep memory small."""
    with netCDF4.Dataset(path) as ds:
        time = _month_starts(ds["time"])
        lat = np.asarray(ds["lat"][:], dtype="float64")
        lon = np.asarray(ds["lon"][:], dtype="float64")
        rows = np.flatnonzero((lat >= lat_range[0]) & (lat <= lat_range[1]))
        variable = ds["sst"]
        if variable.ndim == 4:  # v6 has a singleton depth level
            raw = variable[:, 0, rows.min():rows.max() + 1, :]
        else:
            raw = variable[:, rows.min():rows.max() + 1, :]
        sst = np.ma.filled(raw.astype("float32"), np.nan)
    lat = lat[rows.min():rows.max() + 1]
    if time.has_duplicates or not time.is_monotonic_increasing:
        raise ValueError("Time coordinate is not strictly increasing.")
    order = np.argsort(lat)
    return xr.DataArray(sst[:, order, :], coords={"time": time, "lat": lat[order], "lon": lon}, dims=("time", "lat", "lon"))


def box_mean(sst: xr.DataArray, region: str, weighted: bool = False) -> pd.Series:
    """Mean SST over a region box, inclusive of edges, ignoring missing cells."""
    lat_min, lat_max, lon_min, lon_max = REGIONS[region]
    box = sst.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    if weighted:
        mean = box.weighted(np.cos(np.deg2rad(box.lat)).fillna(0)).mean(("lat", "lon"))
    else:
        mean = box.mean(("lat", "lon"), skipna=True)
    return mean.to_series().rename(region)


def monthly_anomaly(series: pd.Series, start_year: int = 1991, end_year: int = 2020) -> pd.Series:
    """Subtract the calendar-month mean over [start_year, end_year] inclusive."""
    reference = series[(series.index.year >= start_year) & (series.index.year <= end_year)]
    climatology = reference.groupby(reference.index.month).mean()
    if len(climatology) != 12:
        raise ValueError("Reference period does not cover all 12 calendar months.")
    return series - climatology.reindex(series.index.month).to_numpy()


def relative_anomaly(index_anomaly: pd.Series, tropical_anomaly: pd.Series,
                     start_year: int = 1950, end_year: int = 2020) -> pd.Series:
    """Index anomaly minus the 20S-20N tropical-mean anomaly, rescaled per calendar
    month so its variance equals the original index over [start_year, end_year].

    Follows NOAA CPC's RONI (L'Heureux et al. 2024, J. Climate 37, 1197-1211):
    apply it to 3-month running means to obtain RONI. Removing the shared tropical
    warming keeps early decades from being classified as La Nina."""
    difference = index_anomaly - tropical_anomaly
    period = (index_anomaly.index.year >= start_year) & (index_anomaly.index.year <= end_year)
    months = index_anomaly.index.month
    ratio = index_anomaly[period].groupby(months[period]).std() / difference[period].groupby(months[period]).std()
    return difference * ratio.reindex(months).to_numpy()

