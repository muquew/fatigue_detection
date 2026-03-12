from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
FEATURES_DIR = DATA_DIR / "features"
LABELS_DIR = DATA_DIR / "labels"


@dataclass(frozen=True)
class AppConfig:
    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    preview_display_width: int = 960
    preview_display_height: int = 720
    window_size: int = 15
    window_step: int = 3
    low_ear_threshold: float = 0.21
    high_mar_threshold: float = 0.65
    fatigue_confidence_threshold: float = 0.78
    consecutive_alert_threshold: int = 3
    vote_window_size: int = 7
    vote_positive_threshold: float = 0.67
    alert_confidence_threshold: float = 0.82
    alert_vote_ratio_threshold: float = 0.75
    alert_cooldown_seconds: float = 3.0
    state_labels: dict[int, str] = field(
        default_factory=lambda: {0: "Normal", 1: "Fatigue Risk"}
    )
    preview_window_name: str = "Fatigue Detection"
    default_language: str = "zh"
    raw_videos_dir: Path = field(default_factory=lambda: RAW_VIDEOS_DIR)
    feature_output_dir: Path = field(default_factory=lambda: FEATURES_DIR)
    label_output_dir: Path = field(default_factory=lambda: LABELS_DIR)
    window_label_path: Path = field(
        default_factory=lambda: LABELS_DIR / "window_labels.csv"
    )
    window_label_review_path: Path = field(
        default_factory=lambda: LABELS_DIR / "window_labels_review.csv"
    )
    dataset_path: Path = field(default_factory=lambda: DATA_DIR / "dataset.npz")
    training_report_path: Path = field(
        default_factory=lambda: RESULTS_DIR / "training_metrics.json"
    )
    evaluation_report_path: Path = field(
        default_factory=lambda: RESULTS_DIR / "evaluation_metrics.json"
    )
    rule_baseline_report_path: Path = field(
        default_factory=lambda: RESULTS_DIR / "rule_baseline_metrics.json"
    )
    benchmark_report_path: Path = field(
        default_factory=lambda: RESULTS_DIR / "inference_benchmark.json"
    )
    ablation_report_path: Path = field(
        default_factory=lambda: RESULTS_DIR / "feature_ablation.json"
    )
    summary_report_path: Path = field(
        default_factory=lambda: RESULTS_DIR / "experiment_summary.md"
    )
    landmark_model_path: Path = field(
        default_factory=lambda: MODELS_DIR / "face_landmarker.task"
    )
    scaler_path: Path = field(default_factory=lambda: MODELS_DIR / "scaler.pkl")
    torch_model_path: Path = field(default_factory=lambda: MODELS_DIR / "mlp.pth")
    onnx_model_path: Path = field(default_factory=lambda: MODELS_DIR / "mlp.onnx")


def default_config() -> AppConfig:
    return AppConfig()
