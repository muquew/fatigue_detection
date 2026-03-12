from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from common import ROOT_DIR  # noqa: F401

from src.config import default_config
from src.features import FrameFeatures
from src.window_buffer import WINDOW_FEATURE_NAMES, build_window_feature_vector


@dataclass
class FeatureRow:
    frame_index: int
    features: FrameFeatures


@dataclass
class CandidateWindow:
    start_frame: int
    end_frame: int
    score: float


def build_parser() -> argparse.ArgumentParser:
    config = default_config()
    parser = argparse.ArgumentParser(
        description='Generate heuristic window-level label intervals for review or retraining.'
    )
    parser.add_argument('--input-dir', default=str(config.feature_output_dir))
    parser.add_argument('--output', default=str(config.window_label_review_path))
    parser.add_argument('--min-score', type=float, default=1.2)
    parser.add_argument('--fallback-top-ratio', type=float, default=0.15)
    parser.add_argument('--merge-gap', type=int, default=45)
    return parser


def load_rows(csv_path: Path) -> tuple[list[FeatureRow], str, str]:
    rows: list[FeatureRow] = []
    group_id = csv_path.parent.name
    source_video = csv_path.name
    with csv_path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            group_id = row.get('group_id', group_id)
            source_video = row.get('source_video', source_video)
            rows.append(
                FeatureRow(
                    frame_index=int(row['frame_index']),
                    features=FrameFeatures(
                        ear_left=float(row['ear_left']),
                        ear_right=float(row['ear_right']),
                        ear_avg=float(row['ear_avg']),
                        mar=float(row['mar']),
                        pitch=float(row['pitch']),
                        yaw=float(row['yaw']),
                        roll=float(row['roll']),
                    ),
                )
            )
    return rows, group_id, Path(source_video).name


def score_window(vector: list[float], config) -> float:
    index = {name: pos for pos, name in enumerate(WINDOW_FEATURE_NAMES)}
    ear_min = vector[index['ear_min']]
    mar_max = vector[index['mar_max']]
    close_eye_ratio = vector[index['close_eye_ratio']]
    yawn_ratio = vector[index['yawn_ratio']]
    low_ear_streak = vector[index['low_ear_streak']]
    high_mar_streak = vector[index['high_mar_streak']]

    eye_signal = max(0.0, (0.27 - ear_min) / 0.08)
    mouth_signal = max(0.0, (mar_max - 0.10) / 0.20)
    return (
        eye_signal * 1.6
        + close_eye_ratio * 1.4
        + (low_ear_streak / max(config.window_size, 1)) * 1.2
        + mouth_signal * 1.8
        + yawn_ratio * 1.5
        + (high_mar_streak / max(config.window_size, 1)) * 0.8
    )


def is_positive_window(vector: list[float], config, min_score: float) -> bool:
    index = {name: pos for pos, name in enumerate(WINDOW_FEATURE_NAMES)}
    close_eye_ratio = vector[index['close_eye_ratio']]
    low_ear_streak = vector[index['low_ear_streak']]
    yawn_ratio = vector[index['yawn_ratio']]
    mar_max = vector[index['mar_max']]
    ear_min = vector[index['ear_min']]
    score = score_window(vector, config)
    return (
        score >= min_score
        or close_eye_ratio >= 0.25
        or low_ear_streak >= 4
        or yawn_ratio >= 0.10
        or mar_max >= 0.18
        or ear_min <= 0.22
    )


def merge_candidates(candidates: list[CandidateWindow], merge_gap: int) -> list[tuple[int, int, int, float]]:
    if not candidates:
        return []

    merged: list[tuple[int, int, int, float]] = []
    current_start = candidates[0].start_frame
    current_end = candidates[0].end_frame
    current_count = 1
    current_best_score = candidates[0].score

    for candidate in candidates[1:]:
        if candidate.start_frame <= current_end + merge_gap:
            current_end = max(current_end, candidate.end_frame)
            current_count += 1
            current_best_score = max(current_best_score, candidate.score)
            continue
        merged.append((current_start, current_end, current_count, current_best_score))
        current_start = candidate.start_frame
        current_end = candidate.end_frame
        current_count = 1
        current_best_score = candidate.score

    merged.append((current_start, current_end, current_count, current_best_score))
    return merged


def build_positive_candidates(rows: list[FeatureRow], config, min_score: float) -> list[CandidateWindow]:
    all_windows: list[CandidateWindow] = []
    selected: list[CandidateWindow] = []
    for start in range(0, max(len(rows) - config.window_size + 1, 0), config.window_step):
        window_rows = rows[start : start + config.window_size]
        vector = build_window_feature_vector([item.features for item in window_rows], config)
        if vector is None:
            continue
        score = score_window(vector, config)
        candidate = CandidateWindow(
            start_frame=window_rows[0].frame_index,
            end_frame=window_rows[-1].frame_index,
            score=score,
        )
        all_windows.append(candidate)
        if is_positive_window(vector, config, min_score):
            selected.append(candidate)

    if selected:
        return selected

    if not all_windows:
        return []

    fallback_count = max(1, int(len(all_windows) * 0.15))
    ranked = sorted(all_windows, key=lambda item: item.score, reverse=True)
    return ranked[:fallback_count]


def main() -> int:
    args = build_parser().parse_args()
    config = default_config()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    csv_files = sorted(input_dir.rglob('*.csv'))
    if not csv_files:
        print(f'No feature CSV files found under {input_dir}')
        return 1

    records: list[dict[str, object]] = []
    kept_positive_intervals = 0
    kept_positive_windows = 0

    for csv_path in csv_files:
        rows, subject_id, video_name = load_rows(csv_path)
        if not rows:
            continue

        video_key = csv_path.stem.lower()
        first_frame = rows[0].frame_index
        last_frame = rows[-1].frame_index

        if video_key == '0':
            records.append(
                {
                    'subject_id': subject_id,
                    'video_name': video_name,
                    'start_frame': first_frame,
                    'end_frame': last_frame,
                    'label': 0,
                    'source_rule': 'normal_full_range',
                    'num_windows': '',
                    'max_score': '',
                }
            )
            continue

        if video_key != '10':
            continue

        candidates = build_positive_candidates(rows, config, args.min_score)
        merged = merge_candidates(sorted(candidates, key=lambda item: item.start_frame), args.merge_gap)
        for start_frame, end_frame, num_windows, max_score in merged:
            records.append(
                {
                    'subject_id': subject_id,
                    'video_name': video_name,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'label': 1,
                    'source_rule': 'heuristic_positive_interval',
                    'num_windows': num_windows,
                    'max_score': round(max_score, 4),
                }
            )
            kept_positive_intervals += 1
            kept_positive_windows += num_windows

    if not records:
        print('No review label intervals were generated.')
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'subject_id',
        'video_name',
        'start_frame',
        'end_frame',
        'label',
        'source_rule',
        'num_windows',
        'max_score',
    ]
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (str(item['subject_id']), str(item['video_name']), int(item['start_frame']))):
            writer.writerow(record)

    print(
        f'Saved review labels to {output_path} with intervals={len(records)} '
        f'positive_intervals={kept_positive_intervals} positive_windows={kept_positive_windows}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
