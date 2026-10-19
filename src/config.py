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
    # Identifiers and redundant location columns (intersection_id already
    # identifies the location; the timestamp is split into calendar features).
    "identifiers": ["record_id", "road_id", "latitude", "longitude", TIMESTAMP],
}

# Features kept for the model: context known before the hour starts.
CATEGORICAL_FEATURES = [
    "city_zone",
    "intersection_id",
    "road_type",
    "peak_period",
    "weather_condition",
    "road_condition",
    "iot_sensor_health",
    # Calendar values are one-hot encoded so the MLP does not assume that,
    # for example, hour 23 is "far" from hour 0.
    "hour",
    "day_of_week",
]
NUMERICAL_FEATURES = [
    "lanes",
    "speed_limit",
    "temperature",
    "rainfall_mm",
    "visibility_km",
    "parking_occupancy",
    "traffic_camera_count",
]
BINARY_FEATURES = [
    "is_weekend",
    "is_holiday",
    "rush_hour",
    "nearby_school",
    "nearby_hospital",
    "nearby_market",
    "construction_activity",
    "public_event",
    "accident_reported",
    "emergency_vehicle_detected",
]


@dataclass
class TrainConfig:
    train_frac: float = 0.6
    val_frac: float = 0.2  # the remaining 0.2 is the test set
    hidden_sizes: list = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10
    # Weight CrossEntropyLoss by inverse class frequency (computed on the
    # training set) so the rare Moderate/High classes are not ignored.
    class_weighted_loss: bool = True
    # Oversample minority classes so each mini-batch is roughly class-balanced.
    balanced_sampling: bool = False
    # Std of Gaussian noise added to the standardised numerical inputs during
    # training (data augmentation); 0 disables it.
    input_noise: float = 0.0
    seed: int = 42
