# Traffic Congestion Level Classifier

A PyTorch Multi-Layer Perceptron that classifies the congestion level of an
urban intersection (Low / Moderate / High / Severe), trained on the CityFlow
Smart Urban Mobility and Traffic IoT Dataset (Kaggle).

## Setup

```bash
pip install -r requirements.txt
```

The dataset is included as `data/smart_city_traffic_mobility.zip` and is read
directly from the zip. An unzipped `data/smart_city_traffic_mobility.csv` is
used instead if present (the CSV itself is not committed).

## Usage

On Windows, double-click `start.bat`. It installs missing packages on the
first run, trains the model, and keeps the window open so the results or any
error stay visible. (Double-clicking `main.py` directly closes the window as
soon as the script ends.)

To produce every table and figure for the report in one go, double-click
`run_all.bat` (or run the four scripts below in order). The multi-seed
experiments take about an hour on a laptop CPU.

| Output | Contents |
| --- | --- |
| `outputs/metrics.json`, `training_curves.png`, `confusion_matrix.png` | Final model metrics and plots |
| `outputs/data/` | Data summary tables, class distribution, label by hour, single-feature leakage |
| `outputs/errors/` | Error breakdown tables, error rate by hour |
| `outputs/experiments/` | `summary.md` (mean ± std over 3 seeds), `results.csv` (every run), `macro_f1.png` |

From a terminal:

```bash
python src/main.py                 # train, evaluate, save to outputs/
python src/explore.py              # data summary tables and figures -> outputs/data/
python src/experiments.py          # compare variants over 3 seeds -> outputs/experiments/
python src/error_analysis.py       # where the trained model fails -> outputs/errors/
python src/leakage_check.py        # evidence for the feature selection
python src/predict.py --input new_rows.csv --output predictions.csv
```

`main.py` options: `--epochs`, `--patience`, `--batch-size`, `--lr`,
`--dropout`, `--seed`, `--class-weights`, `--balanced-sampling`,
`--input-noise`, `--data`, `--output-dir`.

Outputs written to `outputs/`: `model.pt`, `preprocessor.joblib`,
`metrics.json`, `training_curves.png`, `confusion_matrix.png`.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest                 # all 109 tests, a few minutes on a laptop CPU
python -m pytest -m "not slow"   # skip the short training runs
```

The tests cover every module: loading (zip fallback, bad timestamps),
cleaning (each kind of dirty row, idempotence, row order kept), feature
selection (every column kept or dropped exactly once, no leaking column can
reach the model), the time split (strictly chronological, no hour in two
subsets), preprocessing (fitted on training rows only, missing values filled),
the model, training (early stopping, best-weight restore, every training
option, same seed gives the same run), metrics (ordinal metrics punish distant
errors more), baselines, the block bootstrap, the model comparison, input
validation on messy data, prediction end to end, and that no personal data
reaches the output. Shared fixtures are in `tests/conftest.py`.

## Using the model on real-world data

`predict.py` is built for input that is messier than the training set:

| Problem in the input | What happens |
| --- | --- |
| Column names in another case or spelling (`Weather`, `Zone `, `temp`) | Normalised and mapped to the model's names (`COLUMN_ALIASES` in `input_validation.py`) |
| Numbers as text (`3.5 mm`, `1,200`, `N/A`) | Parsed; unreadable values become missing |
| Flags as `yes`/`no`/`true`/`false` | Converted to 1/0 |
| Category spelling variants (`heavy_rain`, `downtown core`) | Mapped onto the categories seen in training; unknown values become missing |
| Impossible values (outside `VALID_RANGES`) | Treated as missing |
| Missing calendar fields | Rebuilt from a timestamp if one is present |
| Missing columns or values | Filled with the training-set median / most frequent value |
| No hour and no timestamp | Not predicted (`data_quality = insufficient_data`) |
| Semicolon or tab separated, Windows encodings | Detected automatically |

Each output row carries `data_quality` (`ok`, `imputed`, `low_quality` when
more than 5 features had to be filled in, or `insufficient_data`), the number
of filled-in features and the model's confidence.

### Privacy

- **Data minimisation.** Only the model's 25 input columns (and a timestamp)
  are read. Every other column, such as licence plates, driver names, phone
  numbers or GPS coordinates, is dropped on arrival; the report counts how
  many columns looked like personal data but never shows their contents.
- **No input copied to the output.** The prediction file contains only the
  prediction, confidence and quality flag, in the input's row order.
- **Pseudonymised IDs.** `--id-column record_id --hash-ids` adds a keyed hash
  (HMAC-SHA256) of the ID so results can be matched back. The key comes from
  the `PREDICT_ID_SALT` environment variable and is never stored, so the hash
  cannot be reversed by someone who only has the output.
- **Nothing sensitive is logged.** Reports hold counts and column names only.
- **Safe loading.** The model checkpoint is loaded with `weights_only=True`.
  The preprocessor is a pickle file: only load one you trained yourself.
- **Keep real data out of git.** Put it in `data/private/`; that folder,
  prediction files and reports are in `.gitignore`. (The dataset zip in
  `data/` is public, simulated data.)

```bash
PREDICT_ID_SALT=<your secret> python src/predict.py --input data/private/today.csv \
    --output predictions.csv --id-column record_id --hash-ids --report report.json
```

## Project structure

| File | Responsibility |
| --- | --- |
| `src/config.py` | Paths, kept/dropped feature lists, hyperparameters |
| `src/data_loader.py` | Read the CSV, parse timestamps, validate labels |
| `src/cleaning.py` | Drop duplicate / unusable rows, flag impossible values as missing |
| `src/features.py` | Feature sets; select model inputs; fail if any column is unaccounted for |
| `src/split.py` | Chronological 60/20/20 train/validation/test split |
| `src/preprocessing.py` | StandardScaler + one-hot encoding, label encoding |
| `src/dataset.py` | Arrays to PyTorch `DataLoader` (optional class-balanced sampling) |
| `src/model.py` | `CongestionMLP` (`nn.Module`) |
| `src/pipeline.py` | Shared split -> preprocess -> train steps |
| `src/train.py` | Mini-batch training, input-noise augmentation, early stopping |
| `src/baseline.py` | Majority-class and lookup-table baselines |
| `src/evaluate.py` | Prediction and test metrics |
| `src/plots.py` | Training curves and confusion matrix |
| `src/style.py` | Shared figure colours and styling |
| `src/md_table.py` | DataFrame to Markdown table |
| `src/explore.py` | Exploratory data analysis |
| `src/experiments.py` | Multi-seed comparison of modelling choices |
| `src/error_analysis.py` | Error breakdown of the trained model |
| `src/leakage_check.py` | Single-feature predictiveness report |
| `src/main.py` | Runs the full pipeline |
| `src/input_validation.py` | Normalise and validate real-world input; drop personal data |
| `src/predict.py` | Predicts labels for new rows with a saved model; privacy-preserving output |
| `tests/` | pytest suite, one file per module (see Testing below) |

## Method

1. **Cleaning.** Drop exact duplicates, repeated readings for the same
   intersection and hour, and rows without a valid label or timestamp. Values
   outside a plausible range (`VALID_RANGES` in `config.py`) and non-0/1 flags
   are set to missing. The current dataset is clean, so nothing is removed,
   but the step protects against future data. A report is printed and saved in
   `metrics.json`.
2. **Time-based split.** The 85 days of hourly data are cut on unique
   timestamps: train 2026-01-01 to 02-20, validation 02-21 to 03-09, test
   03-10 to 03-26. Every intersection at a given hour falls in the same subset,
   so the model is always evaluated on hours it has not seen.
3. **Preprocessing fitted on train only.** Missing values are filled with the
   training-set median (numerical) or most frequent value (categorical and
   binary). Numerical features then use `StandardScaler`; categorical features (including `hour` and `day_of_week`)
   use one-hot encoding; binary flags pass through unchanged. This gives 72
   inputs.
4. **Model.** Input → 256 → 128 → 64 → 4, with ReLU and Dropout (0.3) after
   each hidden layer.
5. **Training.** Adam (lr 1e-3), batch size 512, unweighted CrossEntropyLoss.
   Weighting the loss by inverse class frequency was tried and made results
   worse (see Results). Training stops when validation loss has not improved
   for 10 epochs, and the best weights are restored.
6. **Evaluation.** Accuracy and macro/weighted precision, recall and F1 on the
   test set. Macro F1 is the primary metric, compared against a majority-class
   baseline.

## Feature selection

The dataset is simulated, and several columns are the congestion outcome
itself or are computed from it. Keeping them would let the model copy the
answer instead of learning anything. `leakage_check.py` fits a depth-4 decision
tree on each column alone. The majority-class rate is 0.498.

| Group (dropped) | Columns | Why |
| --- | --- | --- |
| Label source | `congestion_score` | The label is exactly this score binned at 25 / 55 / 80 (single-feature accuracy 1.000) |
| Same-hour traffic state | `vehicle_count`, `heavy_vehicle_count`, `motorcycle_count`, `public_transport_count`, `traffic_flow_rate`, `average_speed`, `traffic_density`, `queue_length`, `average_wait_time` | Measurements of the congestion being predicted (`traffic_density` alone: 0.960; correlation with the score up to 0.95; `traffic_flow_rate` equals `vehicle_count`) |
| Derived from traffic | `emission_estimate`, `fuel_waste_estimate`, `air_quality_index` | Consequences of congestion (up to 0.906 alone) |
| Signal response | `signal_cycle_seconds`, `green_light_duration` | The simulator adapts signals to congestion (a 120 s cycle occurs only with High/Severe; green time alone: 0.898). It is also what a controller would set *from* this model's output, so as an input it would be circular |
| Identifiers | `record_id`, `road_id`, `latitude`, `longitude`, `timestamp` | IDs, or redundant with `city_zone`, the road attributes and the existing calendar columns |
| No effect | `intersection_id` | Within a zone, intersections differ only by noise. Removing it left test macro F1 unchanged over 3 seeds and cut the input from 172 to 72 dimensions |

Vehicle-mix ratios (heavy / motorcycle / public transport / other as shares of
`vehicle_count`) were also tested as extra inputs. They avoid the total-volume
leak but carry no signal in this simulation (same macro F1), so they are not used.

**Kept:** location and road context (`city_zone`,
`road_type`, `lanes`, `speed_limit`, `traffic_camera_count`, nearby
school/hospital/market), calendar (`hour`, `day_of_week`, `is_weekend`,
`is_holiday`, `rush_hour`, `peak_period`), weather (`temperature`,
`rainfall_mm`, `visibility_km`, `weather_condition`, `road_condition`) and
events (`construction_activity`, `public_event`, `accident_reported`,
`emergency_vehicle_detected`, `iot_sensor_health`, `parking_occupancy`).
These are all known before or at the start of the hour, so the model
*forecasts* congestion instead of describing it after the fact.

## Results (test set)

Final model, seed 42:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- |
| Majority baseline | 0.493 | 0.123 | 0.250 | 0.165 | 0.326 |
| Lookup: hour × weekend × zone | 0.950 | - | - | 0.890 | 0.947 |
| MLP | 0.967 | 0.948 | 0.917 | **0.930** | 0.965 |

Per class F1: Low 0.975, Moderate 0.813, High 0.943, Severe 0.990.

Variants (`experiments.py`, mean of 3 seeds; each row changes one thing
relative to the final model):

| Variant | Accuracy | Macro F1 |
| --- | --- | --- |
| Final model | 0.966 | 0.930 |
| All columns incl. leaking ones | 0.992 | 0.984 |
| Random split instead of time split | 0.967 | 0.931 |
| Class-weighted loss | 0.945 | 0.909 |
| Oversampling minority classes | 0.945 | 0.908 |
| Gaussian input noise (std 0.1) | 0.966 | 0.930 |
| No dropout | 0.965 | 0.929 |
| No dropout and no early stopping (40 epochs) | 0.964 | 0.926 |
| No time features | 0.515 | 0.327 |
| Plus `intersection_id` | 0.966 | 0.929 |
| Plus vehicle-mix ratios | 0.966 | 0.930 |

Seed-to-seed standard deviation is at most 0.004 except for "no time
features" (0.018), so only differences larger than that are meaningful.

### Where the model fails

Half of the remaining errors are in the evening (17–19h), mostly Moderate
predicted as Low. On weekdays at 18–19h the label depends on the zone (Downtown
and Financial District are ~90% Moderate, the other zones ~44%) and on bad
weather, accidents and public events, which push it to Moderate. With clear
weather and no incident, the outer zones are close to a coin flip between Low
and Moderate, and the error rate there is about 32%. The class-weighted model
predicted Moderate for every one of these rows, which is why it was dropped.

### Limitation: the simulator is driven mainly by time of day

`hour` alone predicts the label with 0.88 accuracy, and a lookup table on hour,
weekend and zone reaches 0.950 accuracy. Removing the time features drops macro
F1 to 0.30. Weather and events matter only around the evening transition.
Real traffic is noisier, so these numbers should not be read as real-world
performance.
