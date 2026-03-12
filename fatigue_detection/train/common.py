from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
POSITIVE_LABEL_KEYWORDS = (
    'fatigue',
    'drowsy',
    'sleepy',
    'yawn',
    '疲劳',
    '哈欠',
    '闭眼',
)
UTA_RLDD_LABEL_MAP = {'0': 0, '5': 1, '10': 1}


def project_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT_DIR.resolve())).replace('\\', '/')
    except Exception:
        return str(candidate).replace('\\', '/')


def list_video_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in VIDEO_SUFFIXES else []
    return sorted(
        item
        for item in path.rglob('*')
        if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
    )


def infer_label_from_name(path: Path) -> int:
    stem = path.stem.lower()
    return 1 if any(keyword in stem for keyword in POSITIVE_LABEL_KEYWORDS) else 0


def infer_dataset_name(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    return 'uta-rldd' if 'uta-rldd' in parts else 'generic'


def infer_label(path: Path, dataset_name: str = 'auto') -> int:
    resolved_dataset = infer_dataset_name(path) if dataset_name == 'auto' else dataset_name
    if resolved_dataset == 'uta-rldd':
        key = path.stem.strip().lower()
        if key in UTA_RLDD_LABEL_MAP:
            return UTA_RLDD_LABEL_MAP[key]
    return infer_label_from_name(path)


def infer_group(path: Path, dataset_root: Path | None = None) -> str:
    if dataset_root is not None:
        try:
            relative = path.relative_to(dataset_root)
            if len(relative.parts) > 1:
                return relative.parts[0]
        except ValueError:
            pass
    return path.parent.name or 'default'


def binary_metrics(y_true, y_pred) -> dict[str, float | list[list[int]]]:
    tp = sum(int(pred == 1 and true == 1) for true, pred in zip(y_true, y_pred))
    tn = sum(int(pred == 0 and true == 0) for true, pred in zip(y_true, y_pred))
    fp = sum(int(pred == 1 and true == 0) for true, pred in zip(y_true, y_pred))
    fn = sum(int(pred == 0 and true == 1) for true, pred in zip(y_true, y_pred))
    total = max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / total
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': [[tn, fp], [fn, tp]],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
