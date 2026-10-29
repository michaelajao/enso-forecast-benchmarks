# ENSO forecast benchmarks

Python tools for preparing climate observations and evaluating forecasts of the
El Niño–Southern Oscillation (ENSO). The code supports regional SST indices,
ocean recharge indicators, linear and classical machine-learning baselines,
single- and multi-task networks, and gridded CNN/ConvLSTM models.

## Installation

Use conda. From the project root:

```powershell
conda env create -f environment.yml
conda activate enso-climate
```

The environment definition includes PyTorch and the data-processing dependencies.
It describes a runtime to create outside the source package; no Python interpreter,
conda environment or downloaded libraries are included with the code. Alternatively,
activate an existing conda environment containing these dependencies and install
the source with `python -m pip install -e .`. Choose a PyTorch build appropriate to
your hardware when configuring a GPU environment; CUDA is optional.

## Data and configuration

Input datasets must be obtained separately. The pipeline uses ERSST SST, CCMP
surface winds, ERA5 precipitation, NOAA ocean heat content and ORAS5 depth of the
20 °C isotherm. A separate robustness command uses NOAA PMEL warm water volume.
`configs/data_catalog.toml` records source filenames, product versions, units and quality flags.
Keep original inputs unchanged in `data/raw/`.

Copy `configs/data_sources.example.toml` to
`configs/data_sources.local.toml` for machine-specific paths. The local
configuration is ignored by Git. The inventory command accepts the configured
source; processing commands also require the source layout and filenames defined
in their loaders.

Two preparation steps need particular attention:

- The supplied January–September 2019 wind files are copies of 2016. The build
  command expects verified replacements under
  `data/external/ccmp_v3.0_2019_replacement/`; it checks source dates against the
  requested months. Preserve the original copies.
- ORAS5 downloads require a Copernicus Climate Data Store account, acceptance of
  the dataset licence and credentials in `~/.cdsapirc`. Downloaded grids are
  reduced to regional indices when processing is enabled.

Raw data, processed datasets, downloaded sources and generated forecast outputs
are local files and are not distributed with the code.

## Preparing inputs

Run commands from the project root with the project environment active:

```powershell
python scripts/prepare_data.py inventory
python scripts/prepare_data.py hash
python scripts/prepare_data.py check-grids
python scripts/prepare_data.py oras5-download 1958 2026 --process
python scripts/prepare_data.py build
```

The build command writes monthly and quarterly tables to `data/processed/`.
Additional inputs are generated with `prepare_data.py grids` for SST/D20 maps
and `prepare_data.py long-indices` for the longer SST record. These commands
require their source data to be available locally.

## Running forecasts

Each stage script accepts a command as its first argument. Invoking it without
a command prints the available commands; required dependencies must be installed.
The experiment identifiers below are command names used by the code.

| Script | Commands and purpose |
| --- | --- |
| `scripts/baselines.py` | `lead-lag`: descriptive correlations; `e1`: reference models; `e2`: classical machine learning; `e6`: target/product sensitivity; `e7`: leads to 24 months; `e9`: unequal records; `e20`: seasonal ridge |
| `scripts/oscillator.py` | `e5`: autumn IOD; `e13`: additional recharge/Indian Ocean states; `e14`: Atlantic states; `e15`: probabilistic scores; `e17`: nonlinear terms; `e18`: linear attribution; `e19-ablation`: input ablation; `e21-pmel`: WWV source sensitivity; `operators`: seasonal operators |
| `scripts/networks.py` | `mtl --variant=NAME`: matched index networks; `gridded`: CNN/ConvLSTM; `tune`: gridded hyperparameter selection; `shortcuts`: evaluation-protocol diagnostics |
| `scripts/evaluate.py` | `e6-seasons`: spring/era comparisons; `e9-rules`: unequal-record decision rules; `e10-p3`: selection diagnostics; `e11`: phase scores; `e12`: model combinations; `e16`: event scores; `e22`: multi-task gain bounds; `seasonal-target`: three-month-mean targets |

For example:

```powershell
python scripts/baselines.py e1
python scripts/baselines.py e7
python scripts/networks.py mtl --variant=ro
python scripts/networks.py gridded
```

Network runs accept `--years=A-B` to select forecast-origin years and `--collect`
to combine saved checkpoints. Some variants require earlier forecast files as
references. The tuning command uses stages: `--configs`, `--run=inner`,
`--select`, `--run=outer` and `--collect`. It requires processed maps and saved
reference forecasts; consult its function documentation for worker and GPU options.

The separate 24-month index-network extension uses `mtl --variant=e24_ro` and
`mtl --variant=e24_backbone`. Add `--check-inputs` to check required files or
`--benchmark` to train only the first two origins at the fixed full settings.
It preserves the original 12-month runs in separate checkpoint directories.
`python scripts/evaluate.py e24` scores the extension against its linear references;
add `--with-gridded` when the gridded forecast artifacts have also been restored.

Forecast tables and checkpoints are written under `reports/tables/`, with
generated analysis figures under `reports/figures/`. Output directories are created
when a command runs; empty data and report folders are not stored in Git. Evaluation commands read
saved forecasts and must run after their dependencies have been generated.
If numerical scripts encounter memory-allocation errors on Windows, set
`$env:OPENBLAS_NUM_THREADS="1"` before starting Python.

## Evaluation protocol

- Standard forecasts use chronological rolling origins, with model fitting and
  preprocessing restricted to years before the test origin.
- Training samples are retained only when their target month precedes the test
  cutoff. Quarterly inputs become available after their quarter is complete.
- Network epochs and auxiliary-loss weights use validation years within the
  training period. Regression penalties use chronological cross-validation.
- Comparisons use paired forecasts and moving-block bootstrap intervals over
  years. Different model families can have different maximum forecast leads;
  an unevaluated lead is not a zero score.
- Deliberately non-chronological or test-selected runs are diagnostic protocols
  for measuring inflation and must not be interpreted as valid forecast skill.

Reproducing numerical results requires the corresponding inputs, settings, seeds
and saved run provenance. This source package contains no forecast outputs or
verified numerical performance claims. A missing artifact must be recovered or
regenerated before its dependent scoring command can run.

The current source still needs an evaluation audit of centred RONI labels at
fitting cutoffs and of preprocessing/target availability within inner validation
splits. These limitations must be resolved before describing affected numerical
results as having followed a fully isolated validation protocol. Deliberately
leaky diagnostic variants must remain separate from the primary comparison.

## Code layout

```text
configs/             Source catalogue and example local configuration
scripts/             Data preparation, forecasting and evaluation commands
src/enso_climate/    Reusable data, forecasting, network and plotting functions
data/                Local source and processed datasets
reports/             Inventories and generated analysis outputs
environment.yml      Conda runtime definition
```

## License

The code is provided under the MIT license in `LICENSE`. The license covers this
software, not third-party observations, reanalysis products, pretrained models or
downloaded papers. Obtain those separately under their providers' terms.

## Reproducibility

Record the code revision, package versions, product versions, input checksums,
forecast dates and fixed settings with each run. Keep checkpoints and outputs
separate when the code, protocol or inputs change. This package is a source
release, not an operational climate-forecast service or a complete archive of a
published experiment. A paper citation and archived release identifier should be
added when available.
