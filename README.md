# ENSO forecast benchmarks

Research code for comparing seasonal recharge dynamics, classical machine
learning, deep learning and multi-task learning in forecasts of the
El Niño–Southern Oscillation (ENSO) from observations. The workflow prepares
climate indices and gridded inputs, generates forecasts and evaluates how model
choice and evaluation design affect estimated skill.

## Research paper

The code accompanies the working manuscript:

> **Can classical machine learning, deep learning and multi-task learning outperform a seasonal recharge oscillator in ENSO prediction from observations?**

The manuscript is in preparation. Author names, affiliations and publication
information will be added later. A BibTeX template appears under
[Citing the work](#citing-the-work).

## What the code covers

| Component | Scope |
| --- | --- |
| Physical and statistical references | Seasonal recharge oscillator, linear inverse models, ridge regression and climatology |
| Classical machine learning | Support-vector regression, random forests, gradient boosting and multilayer perceptrons |
| Neural forecasts | Matched single-task and multi-task index networks, CNNs and ConvLSTMs |
| Evaluation | Chronological forecasts, seasonal and lead-time comparisons, paired uncertainty estimates, and diagnostics of test-period selection and random splits |

The repository provides source code and configuration. It does not contain raw
observations, reanalysis grids, processed datasets, forecast tables, trained
weights or the manuscript. Numerical results require the corresponding inputs
and run settings; the source release is not a complete results archive.

## Set up with conda

Clone the repository and create the environment from the project root:

```shell
git clone https://github.com/michaelajao/enso-forecast-benchmarks.git
cd enso-forecast-benchmarks
conda env create -f environment.yml
conda activate enso-climate
```

`environment.yml` includes the data-processing libraries and PyTorch and installs
the local package in editable mode. The environment is created separately from
the source tree. Select a PyTorch/CUDA build appropriate to your hardware if you
intend to use a GPU; CUDA is optional. Alternatively, use an existing conda
environment with the dependencies listed in `environment.yml` and install the
source with `python -m pip install -e .`.

Run the commands below from the repository root with that environment active.

## Obtain the data

**Data are obtained from their original providers, not hosted in this repository.**
The source catalogue in [configs/data_catalog.toml](configs/data_catalog.toml)
records the study's product choices, expected filenames, units and quality flags.
Use those choices when reproducing a run rather than substituting the latest
product version without recording the change.

| Input | Original provider | Use in the workflow |
| --- | --- | --- |
| ERSST sea-surface temperature | [NOAA NCEI](https://www.ncei.noaa.gov/products/extended-reconstructed-sst) | Regional SST indices and gridded SST inputs; ERSST v6 is the selected source |
| CCMP surface winds | [Remote Sensing Systems](https://www.remss.com/measurements/ccmp/) | Monthly wind indices |
| ERA5 precipitation | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means) | Regional precipitation indices |
| Ocean heat content | [NOAA NCEI](https://www.ncei.noaa.gov/access/global-ocean-heat-content/) | Calendar-quarter ocean heat-content indicators |
| ORAS5 ocean reanalysis | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-oras5) | Depth of the 20 °C isotherm and recharge indicators |
| Warm water volume index | [NOAA PMEL](https://pmel.noaa.gov/elnino/upper-ocean-heat-content-and-enso/) | Separate warm water volume source-sensitivity comparison |

Retain the providers' product versions, metadata and licence information. Some
downloads require registration or acceptance of terms. ORAS5 access through the
CDS API requires credentials configured locally in `~/.cdsapirc`.

Keep source files unchanged under `data/raw/`, or configure their local
locations. Copy `configs/data_sources.example.toml` to
`configs/data_sources.local.toml` and set `source_path` and, where needed,
`[source_directories]`. The inventory command accepts the configured source;
processing commands also require the layouts and filenames used by their loaders.
The local configuration and credentials must remain outside version control.

If reusing the study's original data bundle, check the January–September 2019
CCMP files: the original copies duplicated the corresponding 2016 files. The
build step expects verified replacements under
`data/external/ccmp_v3.0_2019_replacement/` and checks their source dates. The
catalogue also records the CCMP version change in 2020.

## Prepare the inputs

After obtaining and arranging the source products, inventory and check them:

```shell
python scripts/prepare_data.py inventory
python scripts/prepare_data.py hash
python scripts/prepare_data.py check-grids
```

Download and reduce the ORAS5 inputs before building the main tables. This
example requests 1958–2026; choose the range needed for your run and check which
months the provider has published:

```shell
python scripts/prepare_data.py oras5-download 1958 2026 --process
python scripts/prepare_data.py build
```

The `--process` option reduces each ORAS5 archive and deletes that downloaded
archive after processing. Preserve a separate archive copy if you need one.
Other raw source files are read without modification.

The build writes `data/processed/enso_monthly.csv` and `enso_quarterly.csv`.
Additional model inputs are prepared separately:

```shell
python scripts/prepare_data.py grids
python scripts/prepare_data.py long-indices
```

`grids` prepares SST/D20 maps for the gridded networks. `long-indices` prepares the
longer SST record for unequal-record comparisons. Both require their source data;
`long-indices` also reads the main processed monthly table.

## Run the experiments

Stage scripts take a command as their first argument. Run a script without a
command to print its command list; dependencies must already be installed.

| Stage script | Main commands |
| --- | --- |
| `scripts/baselines.py` | `e1`: reference models including the seasonal recharge oscillator; `e2`: classical machine learning; `e7`: reference forecasts to 24 months; `e20`: seasonal ridge |
| `scripts/oscillator.py` | Recharge-state extensions, nonlinear terms, input ablations, seasonal operators and `e21-pmel` for WWV source sensitivity |
| `scripts/networks.py` | `mtl --variant=NAME`: matched index networks; `gridded`: CNN/ConvLSTM; `tune`: gridded tuning; `shortcuts`: evaluation diagnostics |
| `scripts/evaluate.py` | Saved-forecast scoring, seasonal and event comparisons, model combinations, multi-task gain bounds and `seasonal-target` scores |

For example, after preparing the required inputs:

```shell
python scripts/baselines.py e1
python scripts/baselines.py e2
python scripts/baselines.py e7
python scripts/networks.py mtl --variant=ro
```

Network commands accept `--years=A-B` to select forecast-origin years and
`--collect` to combine saved checkpoints. Some variants read earlier reference
forecasts. Gridded tuning uses `--configs`, `--run=inner`, `--select`,
`--run=outer` and `--collect`, with processed maps and reference forecasts required.
Consult the command's function documentation for worker and GPU options.

A separate 24-month index-network extension uses `--variant=e24_ro` or
`--variant=e24_backbone`. Inspect its dependencies without starting training:

```shell
python scripts/networks.py mtl --variant=e24_ro --check-inputs
```

For that extension, `--benchmark` trains the first two forecast-origin years at
the fixed full settings. `python scripts/evaluate.py e24` scores saved extension
forecasts; `--with-gridded` additionally requires the gridded forecast artifacts.
The extension uses separate checkpoint directories from the 12-month experiments.

Forecast tables and checkpoints are written under `reports/tables/`; analysis
figures are written under `reports/figures/`. Required output directories are
created when commands run. Evaluation commands must follow the runs that produce
their input forecasts. Empty directories are not stored in Git.

On Windows, if numerical scripts encounter memory-allocation errors, set
`$env:OPENBLAS_NUM_THREADS="1"` in PowerShell before starting the command.

## Evaluation and reproducibility

The primary evaluation is designed around chronological rolling origins,
training-period model fitting and preprocessing, and training-period selection
of network settings and regression penalties. Target months must precede the
fitting cutoff, and quarterly inputs become available after the quarter ends.
Comparisons use paired forecasts and moving-block bootstrap intervals over years.
Maximum forecast leads differ between model families; missing leads do not imply
zero skill.

Random-split, test-selected and other deliberately non-chronological variants are
evaluation diagnostics. Their apparent skill must be reported separately from
the primary chronological comparisons.

An evaluation audit remains necessary for centred RONI labels at fitting cutoffs
and for preprocessing and target availability within inner validation splits.
Affected results should not be described as fully isolated validation until those
checks are resolved.

To reproduce a numerical result, record the code revision, dependency versions,
input versions and checksums, forecast dates, random seeds and fixed settings.
Keep outputs separate when any of these change. Recover or regenerate missing
artifacts before running dependent scoring commands.

## Repository layout

```text
configs/             Source catalogue and example local configuration
scripts/             Data preparation, forecasting, scoring and figure commands
src/enso_climate/    Shared data, forecasting and network implementations
environment.yml      Conda environment definition
pyproject.toml       Package metadata and dependencies
```

Local inputs live under `data/` and generated outputs under `reports/`; both are
ignored by Git. Runtime JSON receipts, checkpoints, local configuration and
credentials are also excluded from version control.

## Citing the work

The entry below is an **unfinished BibTeX template**, not a published-paper
citation. Replace the placeholder author names and `YYYY` before using it.
Add the final publication details and DOI when available; no DOI or publication
venue has been assigned here.

```bibtex
@unpublished{ensoForecastBenchmarks,
  author = {Surname, Given name and Surname, Given name},
  title = {Can classical machine learning, deep learning and multi-task learning outperform a seasonal recharge oscillator in {ENSO} prediction from observations?},
  year  = {YYYY},
  note  = {Manuscript in preparation},
}
```

Place the completed entry in `references.bib`. A minimal LaTeX example is:

```latex
\documentclass{article}
\begin{document}

We use the ENSO forecasting framework described in
\cite{ensoForecastBenchmarks}.

\bibliographystyle{plain}
\bibliography{references}
\end{document}
```

For a published journal article, change the entry type to `@article`, add the
journal, volume, pages or article number and DOI, and replace the draft note.
When identifying the code, include the repository URL and the exact commit or
release used for the analysis.

## License

The software is released under the [MIT License](LICENSE). That licence does not
cover third-party observations, reanalysis products or other downloaded material;
use and cite those under their providers' terms.
