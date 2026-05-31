# CNAS 测试子系统

本目录用于承载第三方机构测试所需的最小闭环，不依赖 GUI、训练脚本或研究实验入口。

## 目录结构

```text
cnas_test/
  manifests/
    full_dataset_v1.json # 当前完整数据集清单
    test_set_v1.json     # 20 张留出测试集清单（历史/抽样口径）
  templates/
    cnas_test_outline_template.md
    cnas_test_report_template.md
  docs/
    CNAS测试大纲_当前版.md
    CNAS测试报告_当前版.md
  runner/
    config.py            # 固定路径与评测参数
    dataset_loader.py    # 测试集加载与样本列表生成
    evaluator.py         # 模型评测与交付产物保存
    report.py            # JSON / Markdown / 交付清单输出
    report_html.py       # 单页 Web 测试报告
    report_docx.py       # Word 测试报告
    provenance.py        # git / 权重 / 测试集 / 环境溯源
    screenshot.py        # 测试过程截图和结果摘要图
    run_eval.py          # 命令行入口
  outputs/
    YYYYMMDD_HHMMSS/
      dataset/           # 临时数据集 YAML 与样本清单
      metrics/           # JSON 结果
      plots/             # 曲线图、混淆矩阵
      screenshots/       # 测试过程截图
      provenance/        # 环境与版本溯源
      reports/           # Markdown / HTML / DOCX 测试报告
      delivery_manifest.json
```

## 设计原则

- 固定测试集
- 固定评测参数
- 固定输出结构
- 与 GUI 和研究训练脚本解耦
- 保留旧入口 `scripts/run_cnas_eval.py` 作为兼容包装层
- `templates/` 作为测试文档模板源
- `docs/` 作为当前版测试大纲/测试报告的结构化源

## 推荐执行方式

```bash
cd /home/bm/Dev/Microlens_DF
OUT_DIR="cnas_test/outputs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
uv run python -m cnas_test.runner.run_eval --save-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/run_console.log"
```

兼容旧命令：

```bash
python scripts/run_cnas_eval.py
```

## 默认参数与默认输出

- 测试集：`cnas_test/manifests/full_dataset_v1.json`
- 测试样本目录：`output/tile_dataset/images/train` 与 `output/tile_dataset/images/val`
- 默认权重：`output/experiments/phase3e/detection_training/b2_nwd_only_phase3e/weights/best.pt`
- 默认输出目录：未指定 `--save-dir` 时自动生成 `cnas_test/outputs/YYYYMMDD_HHMMSS`
- `conf=0.001`
- `iou=0.6`

## 交付物

一次完整执行后，会在本次时间命名目录 `$OUT_DIR` 下生成以下交付物：

- `$OUT_DIR/dataset/cnas_val.yaml`
- `$OUT_DIR/dataset/cnas_val_list.txt`
- `$OUT_DIR/metrics/cnas_eval_results.json`
- `$OUT_DIR/reports/cnas_test_report.md`
- `$OUT_DIR/reports/cnas_test_report.html`
- `$OUT_DIR/reports/cnas_test_report.docx`
- `$OUT_DIR/provenance/provenance.json`
- `$OUT_DIR/screenshots/01_startup_banner.png`
- `$OUT_DIR/screenshots/02_result_summary.png`
- `$OUT_DIR/screenshots/03_metrics_chart.png`
- `$OUT_DIR/delivery_manifest.json`
