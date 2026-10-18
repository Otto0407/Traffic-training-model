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

From a terminal:

```bash
python src/main.py                 # train, evaluate, save to outputs/
python src/leakage_check.py        # evidence for the feature selection
python src/predict.py --input new_rows.csv --output predictions.csv
```

`main.py` options: `--epochs`, `--patience`, `--batch-size`, `--lr`,
`--dropout`, `--seed`, `--no-class-weights`, `--data`, `--output-dir`.

Outputs written to `outputs/`: `model.pt`, `preprocessor.joblib`,
`metrics.json`, `training_curves.png`, `confusion_matrix.png`.

## Project structure

| File | Responsibility |
| --- | --- |
| `src/config.py` | Paths, kept/dropped feature lists, hyperparameters |
| `src/data_loader.py` | Read the CSV, parse timestamps, validate labels |
| `src/features.py` | Select model inputs; fail if any column is unaccounted for |
| `src/split.py` | Chronological 60/20/20 train/validation/test split |
| `src/preprocessing.py` | StandardScaler + one-hot encoding, label encoding |
| `src/dataset.py` | Arrays to PyTorch `DataLoader` |
| `src/model.py` | `CongestionMLP` (`nn.Module`) |
| `src/train.py` | Mini-batch training, per-epoch validation, early stopping |
| `src/baseline.py` | Majority-class baseline |
| `src/evaluate.py` | Prediction and test metrics |
| `src/plots.py` | Training curves and confusion matrix |
| `src/leakage_check.py` | Single-feature predictiveness report |
| `src/main.py` | Runs the full pipeline |
| `src/predict.py` | Predicts labels for new rows with a saved model |

## Method

1. **Time-based split.** The 85 days of hourly data are cut on unique
   timestamps: train 2026-01-01 to 02-20, validation 02-21 to 03-09, test
   03-10 to 03-26. Every intersection at a given hour falls in the same subset,
   so the model is always evaluated on hours it has not seen.
2. **Preprocessing fitted on train only.** Numerical features use
   `StandardScaler`; categorical features (including `hour` and `day_of_week`)
   use one-hot encoding; binary flags pass through unchanged. This gives 172
   inputs.
3. **Model.** Input → 256 → 128 → 64 → 4, with ReLU and Dropout (0.3) after
   each hidden layer.
4. **Training.** Adam (lr 1e-3), batch size 512, CrossEntropyLoss weighted by
   inverse class frequency because Moderate and High are only about 7–8% of
   rows. Training stops when validation loss has not improved for 10 epochs,
   and the best weights are restored.
5. **Evaluation.** Accuracy and macro/weighted precision, recall and F1 on the
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
| Identifiers | `record_id`, `road_id`, `latitude`, `longitude`, `timestamp` | IDs or redundant with `intersection_id` and the calendar features |

**Kept:** location and road context (`city_zone`, `intersection_id`,
`road_type`, `lanes`, `speed_limit`, `traffic_camera_count`, nearby
school/hospital/market), calendar (`hour`, `day_of_week`, `is_weekend`,
`is_holiday`, `rush_hour`, `peak_period`), weather (`temperature`,
`rainfall_mm`, `visibility_km`, `weather_condition`, `road_condition`) and
events (`construction_activity`, `public_event`, `accident_reported`,
`emergency_vehicle_detected`, `iot_sensor_health`, `parking_occupancy`).
These are all known before or at the start of the hour, so the model
*forecasts* congestion instead of describing it after the fact.

## Results (test set, seed 42)

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- |
| Majority baseline | 0.493 | 0.123 | 0.250 | 0.165 | 0.326 |
| MLP | 0.945 | 0.880 | 0.956 | **0.909** | 0.949 |

Per class F1: Low 0.952, Moderate 0.760, High 0.935, Severe 0.988. Most
errors are Low hours predicted as Moderate (precision for Moderate is 0.63),
mainly in the evening hours 18–19, where the two classes are split roughly
50/50 in the data. Early stopping picked epoch 7.

### Limitation: the simulator is driven mainly by time of day

Even without the leaking columns, the task stays fairly easy. `hour` alone
predicts the label with 0.88 accuracy: 00–06 and 20–23 are almost always Low,
and 10–14 are almost always Severe. Weather, events and accidents add little
signal in this simulation. In a quick check with a gradient-boosted tree that
also had `hour`, `rush_hour` and `peak_period` removed, macro F1 fell to about
0.41. So most of the MLP's performance comes from learning the daily pattern
per intersection. Real traffic is noisier, so these numbers should not be read
as real-world performance.
