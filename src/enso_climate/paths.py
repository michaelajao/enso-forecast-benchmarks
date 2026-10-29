"""Project paths shared by the analysis pipeline."""

from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative data paths against the project root."""
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_local_config(path: str | Path | None = None, *, required: bool = True) -> dict:
    """Read untracked local data paths; never change the original inputs."""
    config_path = resolve_project_path(path) if path else PROJECT_ROOT / "configs/data_sources.local.toml"
    if not config_path.is_file():
        if not required:
            return {}
        raise FileNotFoundError(
            f"Local configuration not found: {config_path}. "
            "Copy configs/data_sources.example.toml to configs/data_sources.local.toml "
            "and set source_path."
        )
    with config_path.open("rb") as stream:
        return tomllib.load(stream)


def source_directory(name: str, default: str, legacy_name: str | None = None) -> Path:
    """Resolve a raw-data directory without recording a supplier's personal folder.

    Explicit ``[source_directories]`` paths win. Existing bundles can keep their
    layout: a unique directory with ``legacy_name`` is accepted below data/raw.
    Ambiguous bundles require configuration rather than an arbitrary selection.
    """
    overrides = load_local_config(required=False).get("source_directories", {})
    if name in overrides:
        return resolve_project_path(overrides[name])
    canonical = DATA_DIR / "raw" / default
    if canonical.is_dir() or legacy_name is None:
        return canonical
    matches = sorted(path for path in (DATA_DIR / "raw").rglob(legacy_name) if path.is_dir())
    if len(matches) > 1:
        raise ValueError(
            f"Multiple {name} source directories found. Set source_directories.{name} "
            "in configs/data_sources.local.toml."
        )
    return matches[0] if matches else canonical


def ensure_output_directories() -> None:
    """Create generated-output folders when running a command."""
    for path in (PROCESSED_DIR, REPORTS_DIR / "tables", REPORTS_DIR / "figures"):
        path.mkdir(parents=True, exist_ok=True)
