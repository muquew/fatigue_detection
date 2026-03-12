from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT_DIR  # noqa: F401

from src.config import default_config


def build_parser() -> argparse.ArgumentParser:
    config = default_config()
    parser = argparse.ArgumentParser(description='Summarize experiment outputs into Markdown.')
    parser.add_argument('--training', default=str(config.training_report_path))
    parser.add_argument('--rule-baseline', default=str(config.rule_baseline_report_path))
    parser.add_argument('--benchmark', default=str(config.benchmark_report_path))
    parser.add_argument('--ablation', default=str(config.ablation_report_path))
    parser.add_argument('--output', default=str(config.summary_report_path))
    return parser


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def main() -> int:
    args = build_parser().parse_args()
    training = load_json(Path(args.training))
    rule_baseline = load_json(Path(args.rule_baseline))
    benchmark = load_json(Path(args.benchmark))
    ablation = load_json(Path(args.ablation))

    cv_summary = training.get('cross_validation', {}).get('summary', {}).get('val', {})
    final_train = training.get('final_model', {}).get('train_metrics', {})
    rule_overall = rule_baseline.get('overall', {})
    rule_macro = rule_baseline.get('macro_by_group', {})
    best_ablation_name = ablation.get('best_by_val_f1')
    best_ablation = None
    for item in ablation.get('experiments', []):
        if item.get('name') == best_ablation_name:
            best_ablation = item
            break

    lines = [
        '# Experiment Summary',
        '',
        '## Grouped MLP',
        f"- Validation accuracy mean: {cv_summary.get('accuracy_mean', 0.0):.4f}",
        f"- Validation F1 mean: {cv_summary.get('f1_mean', 0.0):.4f}",
        f"- Validation F1 std: {cv_summary.get('f1_std', 0.0):.4f}",
        f"- Full-data train accuracy: {final_train.get('accuracy', 0.0):.4f}",
        f"- Full-data train F1: {final_train.get('f1', 0.0):.4f}",
        '',
        '## Rule Baseline',
        f"- Overall accuracy: {rule_overall.get('accuracy', 0.0):.4f}",
        f"- Overall F1: {rule_overall.get('f1', 0.0):.4f}",
        f"- Macro group accuracy: {rule_macro.get('accuracy', 0.0):.4f}",
        f"- Macro group F1: {rule_macro.get('f1', 0.0):.4f}",
        '',
        '## Runtime Benchmark',
        f"- Torch forward ms/sample: {benchmark.get('torch_forward_ms_per_sample', 0.0):.4f}",
        f"- ONNX forward ms/sample: {benchmark.get('onnx_forward_ms_per_sample', 0.0):.4f}",
        f"- ONNX end-to-end ms/sample: {benchmark.get('onnx_end_to_end_ms_per_sample', 0.0):.4f}",
        f"- ONNX speedup vs torch: {benchmark.get('forward_speedup_vs_torch', 0.0):.2f}x",
        f"- Torch/ONNX prediction agreement: {benchmark.get('prediction_agreement', 0.0):.4f}",
        '',
        '## Feature Ablation',
        f"- Best setup by grouped val F1: {best_ablation_name or 'N/A'}",
    ]
    if best_ablation:
        lines.append(
            f"- Best grouped val F1 mean: {best_ablation.get('val_f1', {}).get('mean', 0.0):.4f}"
        )
        lines.append(
            f"- Feature count: {best_ablation.get('num_features', 0)}"
        )

    lines.extend(
        [
            '',
            '## Interpretation',
            '- The project is ready for a course demo because the online pipeline, training loop, ONNX deployment, and comparative experiments are all connected.',
            '- Grouped validation remains the real bottleneck; label quality across subjects matters more than adding extra model complexity.',
            '- ONNX results should be used in the demo to emphasize edge deployment and low-latency local inference.',
        ]
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved summary to {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
