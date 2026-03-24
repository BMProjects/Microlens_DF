# CNAS 测试子系统

本目录用于承载第三方机构测试所需的最小闭环，不依赖 GUI、训练脚本或研究实验入口。

## 目录结构

```text
cnas_test/
  manifests/
    test_set_v1.json     # 固定留出测试集清单
  templates/
    cnas_test_outline_template.md
    cnas_test_report_template.md
  docs/
    CNAS测试大纲_当前版.md
    CNAS测试报告_当前版.md
  runner/
    config.py            # 固定路径与评测参数
    dataset_loader.py    # 测试集加载与切片列表生成
    evaluator.py         # 模型评测与交付产物保存
    report.py            # JSON / Markdown / 交付清单输出
    run_eval.py          # 命令行入口
  outputs/
    latest/
      dataset/           # 临时数据集 YAML 与切片清单
      metrics/           # JSON 结果
      plots/             # 曲线图、混淆矩阵
      reports/           # Markdown 测试报告
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
uv run python -m cnas_test.runner.run_eval
```

兼容旧命令：

```bash
python scripts/run_cnas_eval.py
```

## 默认参数与默认输出

- 测试集：`cnas_test/manifests/test_set_v1.json`
- 测试切片目录：`output/tile_dataset/images/val`
- 默认权重：`output/training/stage2_cleaned/weights/best.pt`
- 默认输出目录：`cnas_test/outputs/latest`
- `conf=0.001`
- `iou=0.6`
- 通过标准：`mAP@0.5 >= 0.60`

## 交付物

一次完整执行后，默认会生成以下交付物：

- `cnas_test/outputs/latest/dataset/cnas_val.yaml`
- `cnas_test/outputs/latest/dataset/cnas_val_list.txt`
- `cnas_test/outputs/latest/metrics/cnas_eval_results.json`
- `cnas_test/outputs/latest/reports/cnas_test_report.md`
- `cnas_test/outputs/latest/delivery_manifest.json`
