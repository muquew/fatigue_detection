from __future__ import annotations

import argparse
from collections import deque
from math import ceil
from time import perf_counter

try:
    from .alert import AlertManager
    from .camera import CameraStream
    from .config import default_config
    from .features import FeatureExtractor, FrameFeatures
    from .infer_onnx import OnnxFatigueInferencer
    from .landmarker import FaceLandmarker
    from .ui import StatusOverlay, UIOverlay
    from .window_buffer import FeatureWindowBuffer
except ImportError:
    from alert import AlertManager
    from camera import CameraStream
    from config import default_config
    from features import FeatureExtractor, FrameFeatures
    from infer_onnx import OnnxFatigueInferencer
    from landmarker import FaceLandmarker
    from ui import StatusOverlay, UIOverlay
    from window_buffer import FeatureWindowBuffer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fatigue detection entrypoint.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a synthetic pipeline without camera or model dependencies.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of synthetic frames to process in dry-run mode.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Camera id or video file path for live mode.",
    )
    return parser


def synthetic_features(index: int) -> FrameFeatures:
    if index < 12:
        ear = 0.31
        mar = 0.33
    elif index < 24:
        ear = 0.18
        mar = 0.42
    else:
        ear = 0.21
        mar = 0.78
    return FrameFeatures(
        ear_left=ear,
        ear_right=ear,
        ear_avg=ear,
        mar=mar,
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
    )


def update_vote_state(votes: deque[int], label_id: int, positive_votes_required: int) -> tuple[int, float]:
    votes.append(int(label_id == 1))
    positive_votes = sum(votes)
    vote_ratio = positive_votes / max(len(votes), 1)
    smoothed_label = 1 if positive_votes >= positive_votes_required else 0
    return smoothed_label, vote_ratio


def build_status_message(
    smoothed_label: int,
    vote_ratio: float,
    raw_label: int,
    config,
    language: str = "en",
) -> str:
    messages = {
        "en": {
            "active": "Sustained fatigue signal",
            "transient": "Transient fatigue signal",
            "normal": "Monitoring normally",
        },
        "zh": {
            "active": "持续疲劳信号",
            "transient": "瞬时疲劳信号",
            "normal": "正在稳定监测",
        },
    }
    pack = messages.get(language, messages["en"])
    if smoothed_label == 1:
        return f"{pack['active']} ({vote_ratio:.2f})"
    if raw_label == 1:
        return f"{pack['transient']} ({vote_ratio:.2f})"
    return f"{pack['normal']} ({vote_ratio:.2f})"


def run_dry_mode(frames: int) -> int:
    config = default_config()
    buffer = FeatureWindowBuffer(config)
    inferencer = OnnxFatigueInferencer(
        model_path=config.onnx_model_path,
        scaler_path=config.scaler_path,
        confidence_threshold=config.fatigue_confidence_threshold,
    )
    inferencer.load()
    alert_manager = AlertManager(
        cooldown_seconds=config.alert_cooldown_seconds,
        consecutive_threshold=config.consecutive_alert_threshold,
        confidence_threshold=config.alert_confidence_threshold,
        vote_ratio_threshold=config.alert_vote_ratio_threshold,
    )
    positive_votes_required = max(
        1,
        ceil(config.vote_window_size * config.vote_positive_threshold),
    )
    votes: deque[int] = deque(maxlen=config.vote_window_size)

    start_time = perf_counter()
    last_state = "Warmup"

    for index in range(frames):
        buffer.add(synthetic_features(index))
        vector = buffer.build_feature_vector()
        if vector is None:
            continue

        raw_result = inferencer.predict(vector)
        smoothed_label, vote_ratio = update_vote_state(
            votes,
            raw_result.label_id,
            positive_votes_required,
        )
        state_label = config.state_labels[smoothed_label]
        raw_label_id = (
            raw_result.raw_label_id
            if raw_result.raw_label_id is not None
            else raw_result.label_id
        )
        raw_state_label = config.state_labels[raw_label_id]
        alert = alert_manager.should_alert(
            smoothed_label,
            raw_result.confidence,
            vote_ratio,
        )
        last_state = state_label
        print(
            f"[frame={index:03d}] state={state_label} raw={raw_state_label} "
            f"vote_ratio={vote_ratio:.2f} confidence={raw_result.confidence:.2f} "
            f"source={raw_result.source} alert={alert}"
        )

    elapsed = perf_counter() - start_time
    fps = frames / elapsed if elapsed > 0 else 0.0
    print(f"Dry run complete. last_state={last_state} fps={fps:.2f}")
    return 0


def _resolve_source(source: str | None, default_camera_id: int) -> int | str:
    if source is None:
        return default_camera_id
    if source.isdigit():
        return int(source)
    return source


def run_live_mode(source: str | None) -> int:
    try:
        import cv2
    except ImportError:
        print("OpenCV is required for live mode.")
        return 1

    config = default_config()
    input_source = _resolve_source(source, config.camera_id)
    camera = CameraStream(
        source=input_source,
        width=config.frame_width,
        height=config.frame_height,
    )
    landmarker = FaceLandmarker(config.landmark_model_path)
    feature_extractor = FeatureExtractor(config)
    buffer = FeatureWindowBuffer(config)
    inferencer = OnnxFatigueInferencer(
        model_path=config.onnx_model_path,
        scaler_path=config.scaler_path,
        confidence_threshold=config.fatigue_confidence_threshold,
    )
    alert_manager = AlertManager(
        cooldown_seconds=config.alert_cooldown_seconds,
        consecutive_threshold=config.consecutive_alert_threshold,
        confidence_threshold=config.alert_confidence_threshold,
        vote_ratio_threshold=config.alert_vote_ratio_threshold,
    )
    overlay = UIOverlay(language=config.default_language)
    positive_votes_required = max(
        1,
        ceil(config.vote_window_size * config.vote_positive_threshold),
    )
    votes: deque[int] = deque(maxlen=config.vote_window_size)

    if not camera.open():
        print("Camera could not be opened. Try --dry-run first.")
        return 1

    if not landmarker.load():
        print(f"MediaPipe could not be initialized. {landmarker.last_error}")
        camera.release()
        return 1

    inferencer.load()
    cv2.namedWindow(config.preview_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        config.preview_window_name,
        config.preview_display_width,
        config.preview_display_height,
    )
    previous_tick = perf_counter()

    try:
        while True:
            packet = camera.read()
            if packet is None:
                print("No frame received from camera.")
                break

            now = perf_counter()
            fps = 1.0 / max(now - previous_tick, 1e-6)
            previous_tick = now
            landmarks = landmarker.detect(packet.frame)

            if not landmarks.face_detected:
                buffer.clear()
                votes.clear()
                rendered = overlay.draw(
                    packet.frame,
                    StatusOverlay(
                        state_label="No Face",
                        confidence=0.0,
                        fps=fps,
                        alert=False,
                        backend=landmarks.backend,
                        message="未检测到人脸" if overlay.language == "zh" else "Face not detected",
                    ),
                    landmarks,
                )
                cv2.imshow(config.preview_window_name, rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("l"), ord("L")):
                    overlay.toggle_language()
                if key in (ord("i"), ord("I")):
                    overlay.toggle_details()
                if key in (27, ord("q")):
                    break
                continue

            features = feature_extractor.extract(landmarks)
            if features is None:
                rendered = overlay.draw(
                    packet.frame,
                    StatusOverlay(
                        state_label="Invalid Features",
                        confidence=0.0,
                        fps=fps,
                        alert=False,
                        backend=landmarks.backend,
                        message="特征提取失败" if overlay.language == "zh" else "Feature extraction failed",
                    ),
                    landmarks,
                )
                cv2.imshow(config.preview_window_name, rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("l"), ord("L")):
                    overlay.toggle_language()
                if key in (ord("i"), ord("I")):
                    overlay.toggle_details()
                if key in (27, ord("q")):
                    break
                continue

            buffer.add(features)
            vector = buffer.build_feature_vector()
            if vector is None:
                rendered = overlay.draw(
                    packet.frame,
                    StatusOverlay(
                        state_label="Warmup",
                        confidence=0.0,
                        fps=fps,
                        alert=False,
                        ear=features.ear_avg,
                        mar=features.mar,
                        pitch=features.pitch,
                        yaw=features.yaw,
                        roll=features.roll,
                        backend=landmarks.backend,
                        message=(
                            f"正在收集 {config.window_size} 帧"
                            if overlay.language == "zh"
                            else f"Collecting {config.window_size} frames"
                        ),
                    ),
                    landmarks,
                )
                cv2.imshow(config.preview_window_name, rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("l"), ord("L")):
                    overlay.toggle_language()
                if key in (ord("i"), ord("I")):
                    overlay.toggle_details()
                if key in (27, ord("q")):
                    break
                continue

            raw_result = inferencer.predict(vector)
            smoothed_label, vote_ratio = update_vote_state(
                votes,
                raw_result.label_id,
                positive_votes_required,
            )
            alert = alert_manager.should_alert(
                smoothed_label,
                raw_result.confidence,
                vote_ratio,
            )
            raw_label_id = (
                raw_result.raw_label_id
                if raw_result.raw_label_id is not None
                else raw_result.label_id
            )
            status = StatusOverlay(
                state_label=config.state_labels[smoothed_label],
                confidence=raw_result.confidence,
                fps=fps,
                alert=alert,
                ear=features.ear_avg,
                mar=features.mar,
                pitch=features.pitch,
                yaw=features.yaw,
                roll=features.roll,
                backend=raw_result.source,
                message=build_status_message(
                    smoothed_label,
                    vote_ratio,
                    raw_label_id,
                    config,
                    overlay.language,
                ),
                raw_state_label=config.state_labels[raw_label_id],
                vote_ratio=vote_ratio,
                fatigue_probability=raw_result.fatigue_probability,
            )
            rendered = overlay.draw(packet.frame, status, landmarks)
            cv2.imshow(config.preview_window_name, rendered)

            if alert:
                print(
                    f"Fatigue alert triggered. vote_ratio={vote_ratio:.2f} "
                    f"confidence={raw_result.confidence:.2f} source={raw_result.source}"
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("l"), ord("L")):
                overlay.toggle_language()
            if key in (ord("i"), ord("I")):
                overlay.toggle_details()
            if key in (27, ord("q")):
                break
    finally:
        camera.release()
        landmarker.close()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.dry_run:
        return run_dry_mode(args.frames)
    return run_live_mode(args.source)


if __name__ == "__main__":
    raise SystemExit(main())
