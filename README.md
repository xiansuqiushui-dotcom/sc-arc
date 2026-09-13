# SC-ARC

Reference implementation for **Fitness-for-use assessment of cross-source lithium-ion battery SOH predictions for health-threshold decisions**.

SC-ARC uses five measured cells to assess whether predictions for the remaining cells support health-threshold decisions. This repository provides the method implementation, fixed configuration, a two-stage measurement interface, and deterministic tests.

## Install and test

Python 3.10 is the reference environment.

```console
python -m pip install -r requirements.txt
python test_core.py
```

The tests cover local residual-shift invariance, stable but adverse residuals, threshold equality, strict decision cutoffs, and the five-label interface. They use generated arrays, not experimental battery data.

## Run on your data

Provide historical training records and one processed-feature row per candidate cell. The 35 feature names and fixed method settings are in `config.json`.

```console
python run_new_task.py prepare --history history.csv --features batch_features.csv --dataset-id MY_BATCH --phase 0.25 --out batch_run
```

Measure the five requested cells and fill the `soh` column in `batch_run/measurements_request.csv`, then run:

```console
python run_new_task.py decide --prepared batch_run/prepared_task.npz --measurements batch_run/measurements_request.csv --out batch_run/result
```

The decision stage accepts exactly the requested five labels and does not load the unmeasured cells' outcomes. The configured cutoff was determined using the manuscript's development sources; it is not a universal probability guarantee.

## Input schema

- Historical training CSV: the configured feature columns, `soh`, and record identifiers `source`, `cell_id`, `cycle_number`, `raw_cycle_index`, `file_name`.
- Candidate CSV: `cell_id` and all configured feature columns; exactly one row per cell and no `soh` or `truth` column.
- Measurement CSV: exactly `cell_id,soh`, with the five requested cells, no duplicates or missing values.
- SOH uses a fraction rather than percentage units. Historical and candidate features must have consistent definitions and units.
- `dataset-id` and `phase` control reproducible model randomization. The phase argument is not an estimate of remaining battery life.

## Reproduction scope and data availability

`sc_arc_core.py` implements historical-model training, TreeSHAP selection, current-task predictor selection, residual construction, local scoring and decisions. `run_reproduction.py` contains the original evaluation harness, including source-level cutoff construction. `config.json` retains the original source split and input hashes.

This public release contains **code and configuration only**. Raw battery files, processed experimental features, evaluation labels and reference decision tables are not redistributed here. Their redistribution permissions have not been cleared for this release. Accordingly, `run_reproduction.py` requires the separately obtained original input and reference files listed in `config.json`; a fresh clone does not reproduce the manuscript's numerical tables without those files. The core and two-stage interface can instead run on user-supplied compatible data.

The implementation was copied unchanged from the locally verified core release dated 6 September 2026. Public-release testing checks code behavior, not a new experimental validation. The historical package reported 3,949 correct / 1 incorrect development decisions and 1,823 / 1 external decisions. No additional real-data experiment was run for this upload.

## Files

- `sc_arc_core.py`: method implementation.
- `run_new_task.py`: prepare/measure/decide interface.
- `run_reproduction.py`: original evaluation harness (requires external data).
- `test_core.py`: deterministic unit tests.
- `config.json`: settings, features, source split and input hashes.
- `requirements.txt`: reference dependency versions.

## Authors and permissions

Yuyang Wu, Wei Zuo, Ruijia Yang and Aiping Jiang.

This repository is publicly accessible. No additional software licence is granted in this initial release; third-party dependencies retain their respective licences. Public availability does not grant redistribution rights to third-party battery datasets.
