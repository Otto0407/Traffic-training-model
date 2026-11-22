"""Central configuration: paths, feature lists and hyperparameters."""
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "smart_city_traffic_mobility.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET = "congestion_level"
TIMESTAMP = "timestamp"
CLASS_ORDER = ["Low", "Moderate", "High", "Severe"]

# Columns removed before training, grouped by the reason they are removed.
# See README.md ("Feature selection") for the evidence behind each group.
DROPPED_FEATURES = {
    # The label is a direct binning of congestion_score (<25 Low, <55 Moderate,
    # <80 High, >=80 Severe), so this column is the answer itself.
    "label_source": ["congestion_score"],
    # Measurements of the congestion state at the same hour. In the simulator
    # they are the inputs the score is computed from, so they leak the label.
    "same_hour_traffic_state": [
        "vehicle_count",
        "heavy_vehicle_count",
        "motorcycle_count",
        "public_transport_count",
        "traffic_flow_rate",
        "average_speed",
        "traffic_density",
        "queue_length",
        "average_wait_time",
    ],
    # Quantities derived from the traffic state (consequences of congestion).
    "derived_from_traffic": [
        "emission_estimate",
        "fuel_waste_estimate",
        "air_quality_index",
    ],
    # Signal timing reacts to congestion in the simulator (a 120 s cycle only
    # appears with High/Severe). It is also what a downstream controller would
    # set from our prediction, so using it as an input would be circular.
    "signal_response": ["signal_cycle_seconds", "green_light_duration"],
    # Identifiers and redundant location columns (city_zone and the road
    # attributes describe the location; the dataset already has hour,
    # day_of_week etc., so the raw timestamp is not needed).
    "identifiers": ["record_id", "road_id", "latitude", "longitude", TIMESTAMP],
    # Tested and found to add nothing: within a zone, intersections differ only
    # by noise. Dropping it left test metrics unchanged (3 seeds) and cut the
    # input from 172 to 72 dimensions.
    "no_effect": ["intersection_id"],
    # Removed by feature ablation (src/feature_ablation.py, run on the earlier
    # 25-feature set): retraining without each of these, one at a time and
    # cumulatively, kept validation macro F1 within the seed spread. Most repeat
    # information carried by a kept feature (hour covers peak_period and
    # rush_hour; day_of_week covers is_weekend; weather_condition covers rain,
    # visibility and road condition; city_zone covers the fixed road attributes).
    "redundant_by_ablation": [
        "is_weekend",
        "is_holiday",
        "rush_hour",
        "peak_period",
        "road_type",
        "lanes",
        "speed_limit",
        "traffic_camera_count",
        "temperature",
        "rainfall_mm",
        "visibility_km",
        "road_condition",
        "iot_sensor_health",
        "emergency_vehicle_detected",
        "parking_occupancy",
        "nearby_school",
        # These two were borderline in the ablation; a paired check with five
        # new seeds (100-104) put removing both at -0.0001 ± 0.0008, i.e. noise.
        "nearby_hospital",
        "nearby_market",
    ],
}

# Features kept for the model (7). Note that accident_reported, public_event and
# construction_activity describe the same hour, so the model classifies the
# current hour from its context; it does not forecast ahead.
CATEGORICAL_FEATURES = [
    "city_zone",
    "weather_condition",
    # Calendar values are one-hot encoded so the MLP does not assume that,
    # for example, hour 23 is "far" from hour 0.
    "hour",
    "day_of_week",
]
NUMERICAL_FEATURES = []
BINARY_FEATURES = [
    "construction_activity",
    "public_event",
    "accident_reported",
]


# Physically plausible range for each numerical input (inclusive). Values
# outside it are treated as sensor or entry errors and set to missing during
# cleaning, then filled in by the preprocessor.
VALID_RANGES = {
    "lanes": (1, 12),
    "speed_limit": (5, 200),
    "temperature": (-50, 60),
    "rainfall_mm": (0, 500),
    "visibility_km": (0, 50),
    "parking_occupancy": (0, 100),
    "traffic_camera_count": (0, 50),
    "hour": (0, 23),
    "day_of_week": (0, 6),
}

@dataclass
class TrainConfig:
    train_frac: float = 0.6
    val_frac: float = 0.2  # the remaining 0.2 is the test set
    hidden_sizes: list = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.3
    learning_rate: float = 1e-3
    # "adam" (L2 penalty mixed into the gradient) or "adamw" (decoupled weight
    # decay, the usual choice when weight_decay > 0).
    optimizer: str = "adam"
    weight_decay: float = 0.0
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10
    # Weight CrossEntropyLoss by inverse class frequency (computed on the
    # training set). Off by default: in experiments it made the model predict
    # Moderate for every weekday 18-19h row and lowered macro F1 (0.908 vs 0.929).
    class_weighted_loss: bool = False
    # Oversample minority classes so each mini-batch is roughly class-balanced.
    balanced_sampling: bool = False
    # Std of Gaussian noise added to the standardised numerical inputs during
    # training (data augmentation); 0 disables it.
    input_noise: float = 0.0
    # Reload the weights from the epoch with the lowest validation loss after
    # training. Turned off only to measure what early stopping protects against.
    restore_best_weights: bool = True
    seed: int = 42
