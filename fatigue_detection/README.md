# Fatigue Detection

这是一个边缘智能疲劳检测原型。项目目标不是训练大型视频模型，而是完成一条可本地运行、可解释、可部署的完整闭环：

`摄像头/视频 -> 人脸关键点 -> EAR/MAR/头姿 -> 滑动窗口特征 -> 轻量 MLP -> ONNX Runtime -> 本地界面与告警`

## 文档入口

建议按下面顺序阅读：

1. `docs/01_项目概览.md`
2. `docs/02_使用说明.md`
3. `docs/03_代码结构说明.md`
4. `docs/04_真实演示准备.md`
5. `docs/05_系统详细说明.md`
6. `docs/06_AI视频生成说明.md`
7. `docs/07_一键视频生成方案.md`

## 项目结构

```text
fatigue_detection/
|- data/
|  |- raw_videos/
|  |- features_cv6/
|  `- labels/
|- docs/
|- models/
|- results/
|- src/
`- train/
```

## 环境准备

```bash
conda activate fatigue
pip install -r requirements.txt
```

关键依赖：

- `opencv-python`
- `mediapipe`
- `torch`
- `onnxruntime`
- `scikit-learn`
- `Pillow`

## 最常用命令

链路自检：

```bash
python src/main.py --dry-run --frames 20
```

摄像头实时运行：

```bash
python src/main.py
```

本地视频演示：

```bash
python src/main.py --source path/to/demo.mp4
```

## 运行时按键

- `L`：切换中英文界面
- `I`：展开或收起详细信息
- `Q` / `ESC`：退出

## 当前在线逻辑

在线检测并不是逐帧直接报警，而是三层判断：

1. 每 `15` 帧构成一个时间窗口，并提取 `15` 维窗口特征。
2. ONNX 模型对每个窗口输出疲劳概率，超过阈值后才记为疲劳窗口。
3. 最近 `7` 个窗口再做投票平滑，并叠加连续触发和冷却逻辑，只有持续疲劳信号才会报警。

当前默认阈值见 [docs/05_系统详细说明.md](docs/05_系统详细说明.md)。

## 从零重跑训练与实验

当前保留的是 6 个受试者的最小可复现实验子集。标准流程如下：

```bash
python train/extract_features.py --input data/raw_videos/UTA-RLDD --output-dir data/features_cv6 --frame-stride 5
python train/generate_window_label_review.py --input-dir data/features_cv6 --output data/labels/window_labels_review.csv
python train/build_dataset.py --input-dir data/features_cv6 --window-labels data/labels/window_labels_review.csv --output data/dataset_cv6_review.npz
python train/train_mlp.py --dataset data/dataset_cv6_review.npz --epochs 8 --batch-size 64 --lr 0.001 --seed 42
python train/export_onnx.py
python train/evaluate.py --dataset data/dataset_cv6_review.npz
python train/rule_baseline.py --dataset data/dataset_cv6_review.npz
python train/benchmark_inference.py --dataset data/dataset_cv6_review.npz --num-samples 256
python train/feature_ablation.py --dataset data/dataset_cv6_review.npz --epochs 5 --batch-size 64 --lr 0.001 --seed 42
python train/summarize_results.py
```

## 当前实验结果

数据集：

- `data/dataset_cv6_review.npz`
- 样本数：`10915`
- 特征维度：`15`
- 受试者数：`6`
- 类别分布：`normal=7046`，`fatigue=3869`

分组交叉验证 MLP：

- `accuracy_mean = 0.7238`
- `f1_mean = 0.5680`
- `f1_std = 0.0759`

规则基线：

- `accuracy = 0.7534`
- `f1 = 0.7156`

推理性能：

- `torch_forward = 0.0677 ms/sample`
- `onnx_forward = 0.0215 ms/sample`
- `onnx_end_to_end = 0.1664 ms/sample`
- `onnx_speedup = 3.14x`

特征消融：

- 最优组合：`eye_mouth_window`
- 最优平均 F1：`0.6167`

## 关键输出文件

模型：

- `models/mlp.pth`
- `models/scaler.pkl`
- `models/mlp.onnx`

数据：

- `data/features_cv6/`
- `data/labels/window_labels_review.csv`
- `data/dataset_cv6_review.npz`

结果：

- `results/training_metrics.json`
- `results/evaluation_metrics.json`
- `results/rule_baseline_metrics.json`
- `results/inference_benchmark.json`
- `results/feature_ablation.json`
- `results/experiment_summary.md`

## 说明

- 当前结果全部由 `seed=42` 重新训练得到。
- 训练评估采用受试者分组交叉验证，不是随机混合切分。
- 当前部署模型仍可继续按演示效果微调阈值和 UI。
- 详细代码逻辑、参数说明和触发逻辑见 `docs/05_系统详细说明.md`。
