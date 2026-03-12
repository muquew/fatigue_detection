from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from common import infer_group, infer_label, list_video_files

from src.config import default_config
from src.features import FRAME_FEATURE_NAMES, FeatureExtractor
from src.landmarker import FaceLandmarker


def build_parser() -> argparse.ArgumentParser:
    config = default_config()
    default_input = config.raw_videos_dir / 'UTA-RLDD'
    parser = argparse.ArgumentParser(description='Extract per-frame fatigue features.')
    parser.add_argument(
        '--input',
        default=str(default_input if default_input.exists() else config.raw_videos_dir),
        help='Input video file or directory.',
    )
    parser.add_argument(
        '--output-dir',
        default=str(config.feature_output_dir),
        help='Directory for extracted frame feature CSV files.',
    )
    parser.add_argument(
        '--label',
        type=int,
        choices=(0, 1),
        default=None,
        help='Override label for a single file or all files.',
    )
    parser.add_argument(
        '--dataset-name',
        choices=('auto', 'generic', 'uta-rldd'),
        default='auto',
        help='Dataset preset used to infer labels from filenames.',
    )
    parser.add_argument(
        '--frame-stride',
        type=int,
        default=5,
        help='Only process every Nth frame to speed up extraction.',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip videos whose feature CSV already exists.',
    )
    return parser


def resolve_output_path(video_path: Path, dataset_root: Path, output_dir: Path) -> Path:
    if dataset_root.is_file():
        parent_name = video_path.parent.name or 'root'
        return output_dir / parent_name / f'{video_path.stem}.csv'
    relative = video_path.relative_to(dataset_root)
    return (output_dir / relative).with_suffix('.csv')


def extract_single_video(
    video_path: Path,
    dataset_root: Path,
    output_dir: Path,
    label_override: int | None,
    dataset_name: str,
    frame_stride: int,
    skip_existing: bool,
) -> tuple[int, int, Path]:
    config = default_config()
    output_path = resolve_output_path(video_path, dataset_root, output_dir)
    if skip_existing and output_path.exists():
        return 0, 0, output_path

    landmarker = FaceLandmarker(config.landmark_model_path)
    if not landmarker.load():
        raise RuntimeError(f'Failed to initialize MediaPipe landmarker: {landmarker.last_error}')
    extractor = FeatureExtractor(config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    label = infer_label(video_path, dataset_name) if label_override is None else label_override
    group = infer_group(video_path, dataset_root)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        landmarker.close()
        raise RuntimeError(f'Could not open video: {video_path}')

    saved_rows = 0
    frame_count = 0
    fieldnames = [
        'frame_index',
        'timestamp_seconds',
        'label',
        'dataset_name',
        'group_id',
        'source_video',
        'backend',
        *FRAME_FEATURE_NAMES,
    ]
    try:
        with output_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            while True:
                success, frame = capture.read()
                if not success:
                    break

                if frame_count % max(frame_stride, 1) != 0:
                    frame_count += 1
                    continue

                landmarks = landmarker.detect(frame)
                features = extractor.extract(landmarks)
                if features is None:
                    frame_count += 1
                    continue

                row = {
                    'frame_index': frame_count,
                    'timestamp_seconds': frame_count / max(fps, 1.0),
                    'label': label,
                    'dataset_name': dataset_name,
                    'group_id': group,
                    'source_video': str(video_path.relative_to(dataset_root.parent if dataset_root.is_dir() else dataset_root.parent)),
                    'backend': landmarks.backend,
                    **features.as_dict(),
                }
                writer.writerow(row)
                saved_rows += 1
                frame_count += 1
    finally:
        capture.release()
        landmarker.close()

    return frame_count, saved_rows, output_path


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    video_files = list_video_files(input_path)
    if not video_files:
        print(f'No video files found under {input_path}')
        return 1

    failures = 0
    for video_path in video_files:
        try:
            total_frames, saved_rows, output_path = extract_single_video(
                video_path=video_path,
                dataset_root=input_path,
                output_dir=output_dir,
                label_override=args.label,
                dataset_name=args.dataset_name,
                frame_stride=args.frame_stride,
                skip_existing=args.skip_existing,
            )
            print(
                f'{video_path.name}: total_frames={total_frames} extracted_rows={saved_rows} '
                f'output={output_path}'
            )
        except Exception as exc:
            failures += 1
            print(f'FAILED {video_path}: {exc}')

    if failures:
        print(f'Feature extraction completed with failures={failures}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
