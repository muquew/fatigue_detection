from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import ROOT_DIR  # noqa: F401

from src.config import default_config
from src.features import FrameFeatures
from src.window_buffer import WINDOW_FEATURE_NAMES, build_window_feature_vector


@dataclass
class FeatureRow:
    frame_index: int
    features: FrameFeatures


@dataclass(frozen=True)
class WindowLabelRule:
    subject_id: str
    video_name: str
    start_frame: int
    end_frame: int
    label: int

    def matches(self, subject_id: str, video_name: str, start_frame: int, end_frame: int) -> bool:
        return (
            self.subject_id == subject_id
            and self.video_name == video_name
            and start_frame >= self.start_frame
            and end_frame <= self.end_frame
        )


def build_parser() -> argparse.ArgumentParser:
    config = default_config()
    parser = argparse.ArgumentParser(description='Build training dataset from CSVs.')
    parser.add_argument(
        '--input-dir',
        default=str(config.feature_output_dir),
        help='Directory containing per-video feature CSV files.',
    )
    parser.add_argument(
        '--output',
        default=str(config.dataset_path),
        help='Output NPZ dataset path.',
    )
    parser.add_argument(
        '--window-labels',
        default=str(config.window_label_path),
        help='CSV file containing window-level labels. If absent, falls back to video labels.',
    )
    return parser


def normalize_video_name(value: str) -> str:
    return Path(value.replace('\\', '/')).name.lower()


def load_window_labels(path: Path) -> list[WindowLabelRule]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    rules: list[WindowLabelRule] = []
    with path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            rules.append(
                WindowLabelRule(
                    subject_id=row['subject_id'].strip(),
                    video_name=normalize_video_name(row['video_name'].strip()),
                    start_frame=int(row['start_frame']),
                    end_frame=int(row['end_frame']),
                    label=int(row['label']),
                )
            )
    return rules


def resolve_window_label(
    rules: list[WindowLabelRule],
    subject_id: str,
    source_video: str,
    start_frame: int,
    end_frame: int,
) -> int | None:
    video_name = normalize_video_name(source_video)
    for rule in rules:
        if rule.matches(subject_id, video_name, start_frame, end_frame):
            return rule.label
    return None


def load_feature_rows(csv_path: Path) -> tuple[list[FeatureRow], int, str, str]:
    rows: list[FeatureRow] = []
    label = 0
    group_id = 'default'
    source_video = csv_path.name
    with csv_path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = int(row['label'])
            group_id = row.get('group_id', 'default')
            source_video = row.get('source_video', csv_path.name)
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
    return rows, label, group_id, source_video


def main() -> int:
    args = build_parser().parse_args()
    config = default_config()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    csv_files = sorted(input_dir.rglob('*.csv'))
    if not csv_files:
        print(f'No feature CSV files found under {input_dir}')
        return 1

    window_rules = load_window_labels(Path(args.window_labels))
    use_window_labels = len(window_rules) > 0

    samples: list[list[float]] = []
    labels: list[int] = []
    sources: list[str] = []
    groups: list[str] = []
    label_sources: list[str] = []

    for csv_path in csv_files:
        rows, video_label, group_id, source_video = load_feature_rows(csv_path)
        for start in range(0, max(len(rows) - config.window_size + 1, 0), config.window_step):
            window_rows = rows[start : start + config.window_size]
            vector = build_window_feature_vector(
                [item.features for item in window_rows],
                config,
            )
            if vector is None:
                continue

            start_frame = window_rows[0].frame_index
            end_frame = window_rows[-1].frame_index
            if use_window_labels:
                label = resolve_window_label(
                    window_rules,
                    group_id,
                    source_video,
                    start_frame,
                    end_frame,
                )
                if label is None:
                    continue
                label_source = 'window'
            else:
                label = video_label
                label_source = 'video'

            samples.append(vector)
            labels.append(label)
            sources.append(source_video)
            groups.append(group_id)
            label_sources.append(label_source)

    if not samples:
        if use_window_labels:
            print('No labeled windows matched the current CSV features and window_labels.csv rules.')
        else:
            print('No windowed samples could be created. Check feature CSV contents.')
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        X=np.asarray(samples, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        feature_names=np.asarray(WINDOW_FEATURE_NAMES, dtype='<U32'),
        sources=np.asarray(sources, dtype='<U256'),
        groups=np.asarray(groups, dtype='<U64'),
        label_sources=np.asarray(label_sources, dtype='<U16'),
    )
    print(
        f'Saved dataset to {output_path} with samples={len(samples)} '
        f'feature_dim={len(samples[0])} groups={len(set(groups))} '
        f'label_mode={"window" if use_window_labels else "video"}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
