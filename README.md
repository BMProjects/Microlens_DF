# Microlens_DF

> **Dark-field defect detection, analysis & wear-grading for defocused micro-structured lenses**
> 暗场离焦微结构镜片缺陷检测、分析与磨损评估系统

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-mid--term%20·%20architecture%20stable-success.svg)](#)

From a dark-field micrograph, the system **detects three defect classes (scratch / spot / critical) → analyzes them at the pixel level → produces a quantitative WearScore and an A/B/C/D grade**, and ships a graphical workbench for non-expert operators.

![GUI demo: upload → detect → results](doc/assets/generated/gui_demo.webp)

*Upload a single lens image and get, in one click, the detection overlay, defect topography map, WearScore breakdown, and a wear-grade card.*

---

## ✨ Capabilities

| Capability | Notes |
|---|---|
| Dark-field preprocessing pipeline | Background-template fusion, highlight-ring registration, brightness correction, ROI extraction |
| Defect detection | YOLOv12m + SAHI tiled inference + IOS cross-tile merging; mAP@0.5 = **0.6765** (CNAS test set) |
| Defect segmentation | HRNet-OCR / SegFormer pixel-level morphology (now the core research direction) |
| Wear scoring | WearScore five-factor weighted score → A/B/C/D grade and verdict |
| Graphical workbench | Gradio four-step flow for non-expert operators |
| Third-party testing | Independent, reproducible CNAS accuracy-testing subsystem |

---

## 🏗️ Modular Architecture

The system is organized into 8 modules with clean interfaces (see [System Architecture](doc/系统架构_模块化梳理_20260531.md)):

```text
raw dark-field image ─①dataset─▶ ②preprocessing ─▶ ④detection (YOLO+SFE+NWD) ─┐
                                       └─▶ ⑤segmentation (HRNet-OCR/SegFormer) ─┤
                                                                                ▼
                                       ⑥evaluation (mAP / WearScore grade) ─▶ ⑦frontend
                                                                                ▲
                  ③semi-supervised labeling (pseudo/weak) ──feedback──▶ ①dataset│
                                                              ⑧CNAS third-party testing
```

| # | Module | Key code |
|---|---|---|
| ① | Dataset | `output/dataset_v2/` (data, not tracked), `src/.../data/`, `scripts/build_tile_dataset.py` |
| ② | Preprocessing | `src/darkfield_defects/preprocessing/` |
| ③ | Semi-supervised labeling | `scripts/step1_label_cleanup.py` … `step5_retrain.py`, `generate_weak_masks.py` |
| ④ | Detection | `src/darkfield_defects/{detection,ml}/`, `scripts/infer_full_pipeline.py` |
| ⑤ | Segmentation | `src/.../ml/{segmentation_factory,hrnet_ocr}.py`, `configs/segmentation/mmseg/` |
| ⑥ | Evaluation | `src/darkfield_defects/{eval,scoring}/` |
| ⑦ | Frontend | `src/darkfield_defects/viz/app.py`, `app_services/inference_service.py` |
| ⑧ | Third-party testing | `cnas_test/` |

![System architecture](doc/assets/generated/project_system_architecture.png)

> **Project status (mid-term):** the architecture is stable. The core algorithm focus is shifting from
> the detection baseline toward the **segmentation** pipeline for finer, pixel-level defect analysis.

---

## 🚀 Quick Start

The project uses [`uv`](https://github.com/astral-sh/uv) to manage the Python environment.

```bash
# 1. Install dependencies (the GUI line usually only needs this)
uv sync --extra dev --extra ml --extra viz

# 2. Launch the graphical workbench → http://127.0.0.1:7860
uv run python -m darkfield_defects.viz.app

# 3. Run tests
uv run pytest -q
```

> **Note:** datasets and training weights are not shipped in this repository. The GUI's default weight
> path is `output/experiments/phase3e/.../weights/best.pt`; provide weights locally before running inference.

### Other entry points

```bash
uv run python -m cnas_test.runner.run_eval --help        # CNAS third-party testing
uv run python scripts/train_msd_segmentation.py --help    # segmentation experiments
uv run python scripts/infer_full_pipeline.py --help       # full-image inference pipeline
```

---

## 📂 Layout

```text
src/            source code (8 modules)
scripts/        research & batch scripts
configs/        detection / segmentation configs
tests/          automated tests
cnas_test/      standalone third-party testing subsystem
research_loops/ automated R&D rules and experiment programs
doc/            illustrated web docs + doc/assets/ long-lived reference images
output/         datasets / weights / experiment & run artifacts (not tracked by default)
```

---

## 🔒 Data / Code Separation

The repository tracks **only source, configs, tests, lightweight manifests/templates, and illustrated docs**.
The following are **excluded from version control** by default (see [.gitignore](.gitignore)):

- Datasets and tiles (`output/dataset_v2/`, `output/tile_dataset/`, …)
- Training outputs and experiment artifacts (`output/training/`, `output/experiments/`, `runs/`)
- Model weights (`*.pt` / `*.pth` / `*.onnx`)
- Test-run artifacts (`cnas_test/outputs/`)
- Binary / sensitive deliverables (`*.pdf` / `*.docx` / `*.odt`) and archived docs (`output/_doc_archive/`)

---

## 📑 Documentation

Open [`doc/index.html`](doc/index.html) to browse all illustrated docs. Key documents:

- [System Architecture (modular overview)](doc/系统架构_模块化梳理_20260531.md)
- [Software Design Document (SDD v1)](doc/软件系统说明文档_SDD_v1.md)
- [Research & Tech Note — pre-release & algorithm evolution](doc/研究技术文档_20260324_预发布与算法演进.md)
- [Background Technology Survey v1](doc/背景技术综合研究报告_v1.md)
- [GitHub collaboration & large-file management](doc/GitHub协作同步与大文件管理说明_20260324.md)

---

## 📜 License & IP

The code is licensed under [Apache License 2.0](LICENSE). This project is part of research conducted under
national key R&D program topics (2024YFC2419504 / 2024YFC2419500); the raw data, official task statements,
and sci-tech reports are not published with this repository. See [NOTICE](NOTICE).

> Third-party component Ultralytics YOLO is licensed under AGPL-3.0; if you redistribute or provide a network
> service, comply with its obligations.
