"""单页 Web (HTML) 测试报告生成。

特点：
- 自包含：图片以 base64 内联，可单文件分发给评测专家
- 内容覆盖：测试概况 / 数据集 / 评测参数 / 指标 / 截图 / 溯源
"""

# ruff: noqa: E501

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

from cnas_test.runner.config import (
    DATASET_CONSTRUCTION_METHOD,
    DATASET_NAME,
    DATASET_SUMMARY,
    MODEL_DESCRIPTION,
    RECOMMENDED_COMMAND,
    SOFTWARE_NAME,
    SOFTWARE_VERSION,
    STANDARD_SAMPLE_UNIT,
    VERSION_NOTE,
)


def _img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(suffix, "png")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def _img_tag(path: Path, alt: str) -> str:
    src = _img_b64(path)
    if not src:
        return (
            f'<p class="missing">[缺图] {html.escape(alt)} — 期望路径：{html.escape(str(path))}</p>'
        )
    return f'<figure><img src="{src}" alt="{html.escape(alt)}"/><figcaption>{html.escape(alt)}</figcaption></figure>'


def build_html_report(
    *,
    payload: dict[str, Any],
    provenance: dict[str, Any],
    screenshots: dict[str, Path],
    plots_dir: Path,
    test_set_path: Path,
    weights_path: Path,
    save_path: Path,
) -> Path:
    metrics = payload["metrics"]
    dataset = payload["dataset"]
    dist = dataset["class_counts"]

    ult = plots_dir / "ultralytics_output"
    pr_curve = ult / "BoxPR_curve.png"
    cm = ult / "confusion_matrix_normalized.png"
    pred_samples = [ult / f"val_batch{i}_pred.jpg" for i in range(3)]

    git = provenance.get("git", {})
    weights_info = provenance.get("weights", {})
    test_set_info = provenance.get("test_set", {})
    env = provenance.get("environment", {})

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<title>{html.escape(SOFTWARE_NAME)} CNAS 测试报告</title>
<style>
body {{ font-family: "Noto Sans CJK SC","Source Han Sans SC","Microsoft YaHei",sans-serif; max-width: 1080px; margin: 32px auto; padding: 0 24px; color: #222; line-height: 1.6; }}
h1 {{ border-bottom: 3px solid #4C78A8; padding-bottom: 8px; }}
h2 {{ border-left: 4px solid #4C78A8; padding-left: 10px; margin-top: 32px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #bbb; padding: 6px 10px; text-align: left; }}
th {{ background: #f3f6fb; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
figure {{ margin: 12px 0; text-align: center; }}
figure img {{ max-width: 100%; border: 1px solid #ddd; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
figcaption {{ color: #555; font-size: 0.9em; margin-top: 4px; }}
code, pre {{ font-family: "JetBrains Mono","Source Code Pro",monospace; background: #f7f7f7; }}
pre {{ padding: 10px; overflow-x: auto; border: 1px solid #e3e3e3; }}
.formula {{ margin: 10px 0; padding: 10px 12px; text-align: center; background: #f7f7f7; border: 1px solid #e3e3e3; font-family: "Cambria Math","Times New Roman",serif; font-size: 1.06em; }}
.kv {{ background: #fafbfd; padding: 8px 12px; border-left: 3px solid #888; }}
.missing {{ color: #b22222; }}
</style>
</head>
<body>
<h1>{html.escape(SOFTWARE_NAME)} CNAS 测试报告</h1>
<p class="kv">第三方测试结果记录。测试时间：<b>{html.escape(payload["test_timestamp"])}</b>　|
mAP50 = <b>{metrics["mAP50"]:.4f}</b></p>
<p>本报告用于客观呈现测试环境、测试数据、测试过程、复现条件和实测指标。符合性判定依据以委托测试文件或测试机构正式规则为准。</p>

<h2>1. 测试概况</h2>
<table>
<tr><th>项目</th><th>内容</th></tr>
<tr><td>被测软件名称</td><td>{html.escape(SOFTWARE_NAME)}</td></tr>
<tr><td>测试版本</td><td><code>{html.escape(SOFTWARE_VERSION)}</code></td></tr>
<tr><td>版本备注</td><td>{html.escape(VERSION_NOTE)}</td></tr>
<tr><td>模型描述</td><td>{html.escape(MODEL_DESCRIPTION)}</td></tr>
<tr><td>标准命令</td><td><code>{html.escape(RECOMMENDED_COMMAND)}</code></td></tr>
<tr><td>测试集清单</td><td><code>{html.escape(str(test_set_path))}</code></td></tr>
<tr><td>测试集 SHA256</td><td><code>{html.escape(test_set_info.get("sha256") or "-")}</code></td></tr>
<tr><td>模型权重</td><td><code>{html.escape(str(weights_path))}</code></td></tr>
<tr><td>权重 SHA256</td><td><code>{html.escape(weights_info.get("sha256") or "-")}</code></td></tr>
<tr><td>权重大小</td><td>{weights_info.get("size_bytes", "-")} 字节</td></tr>
</table>

<h2>2. 数据集说明</h2>
<table>
<tr><th>项目</th><th>内容</th></tr>
<tr><td>数据集名称</td><td>{html.escape(DATASET_NAME)}</td></tr>
<tr><td>数据集类型</td><td>{html.escape(STANDARD_SAMPLE_UNIT)}数据集</td></tr>
<tr><td>样本构建方法</td><td>{html.escape(DATASET_CONSTRUCTION_METHOD)}</td></tr>
<tr><td>样本图像尺寸</td><td>640 × 640 像素</td></tr>
<tr><td>样本图像数量</td><td class="num">{dataset["tiles"]}</td></tr>
<tr><td>实际参与评测样本</td><td class="num">{payload["n_tiles"]}</td></tr>
<tr><td>标注缺陷框数量</td><td class="num">{dataset["boxes"]}</td></tr>
<tr><td>空背景样本</td><td class="num">{dataset["background_tiles"]}</td></tr>
<tr><td>类别分布</td><td>scratch={dist["scratch"]}，spot={dist["spot"]}，critical={dist["critical"]}</td></tr>
</table>

<h2>3. 数据划分与执行过程</h2>
<p>单次测试使用固定模型权重和经确认的测试集清单，纳入全部 {payload["n_tiles"]} 个标准化图像样本进行评测。测试程序自动生成数据清单、评测日志、过程截图、指标结果、溯源文件和本报告。</p>
<p>训练后测试先按既定训练配置重新训练模型，再使用训练输出的 <code>weights/best.pt</code> 按单次测试流程评测。训练阶段使用以下固定划分，最终报告指标以测试命令对全量确认样本重新计算得到的结果为准。</p>
<table>
<tr><th>子集</th><th>图像编号数</th><th>标准化图像样本数</th><th>标注缺陷框数</th><th>用途</th></tr>
<tr><td>训练子集</td><td class="num">{DATASET_SUMMARY["train"]["images"]}</td><td class="num">{DATASET_SUMMARY["train"]["tiles"]}</td><td class="num">{DATASET_SUMMARY["train"]["boxes"]}</td><td>参数学习</td></tr>
<tr><td>验证子集</td><td class="num">{DATASET_SUMMARY["val"]["images"]}</td><td class="num">{DATASET_SUMMARY["val"]["tiles"]}</td><td class="num">{DATASET_SUMMARY["val"]["boxes"]}</td><td>训练过程监控与模型选择</td></tr>
</table>

<h2>4. 评测参数</h2>
<table>
<tr><th>参数</th><th>取值</th><th>说明</th></tr>
<tr><td>置信度阈值</td><td class="num">{payload["eval_conf"]}</td><td>低阈值以画全 PR 曲线</td></tr>
<tr><td>NMS IoU</td><td class="num">{payload["eval_iou"]}</td><td>预测框去重叠</td></tr>
<tr><td>AP 匹配 IoU</td><td class="num">0.50</td><td>mAP50 命名一致</td></tr>
</table>

<h2>5. 结果计算方法</h2>
<p>对每个类别 <i>c</i>，预测框按置信度从高到低排序，并在 <i>IoU = 0.5</i> 条件下与同类别标注框进行一对一匹配。</p>
<div class="formula">Precision<sub>c</sub>(k) = TP<sub>c</sub>(k) / [TP<sub>c</sub>(k) + FP<sub>c</sub>(k)]</div>
<div class="formula">Recall<sub>c</sub>(k) = TP<sub>c</sub>(k) / N<sub>gt,c</sub></div>
<div class="formula">AP50<sub>c</sub> = ∫<sub>0</sub><sup>1</sup> P<sub>c</sub>(R)dR，IoU = 0.5</div>
<div class="formula">mAP@0.5 = (1 / C) Σ<sub>c=1</sub><sup>C</sup> AP50<sub>c</sub>，C = 3</div>

<h2>6. 测试结果</h2>
<table>
<tr><th>指标</th><th>数值</th></tr>
<tr><td>scratch AP@0.5</td><td class="num">{metrics["per_class_AP50"]["scratch"]:.4f}</td></tr>
<tr><td>spot AP@0.5</td><td class="num">{metrics["per_class_AP50"]["spot"]:.4f}</td></tr>
<tr><td>critical AP@0.5</td><td class="num">{metrics["per_class_AP50"]["critical"]:.4f}</td></tr>
<tr><td><b>mAP@0.5</b></td><td class="num"><b>{metrics["mAP50"]:.4f}</b></td></tr>
<tr><td>mAP@0.5:0.95</td><td class="num">{metrics["mAP50_95"]:.4f}</td></tr>
<tr><td>Precision</td><td class="num">{metrics["precision"]:.4f}</td></tr>
<tr><td>Recall</td><td class="num">{metrics["recall"]:.4f}</td></tr>
<tr><td>耗时（秒）</td><td class="num">{payload["elapsed_seconds"]:.1f}</td></tr>
</table>

<h2>7. 测试过程截图</h2>
{_img_tag(screenshots.get("startup", Path("-")), "截图 1：测试启动确认横幅")}
{_img_tag(screenshots.get("result_text", Path("-")), "截图 2：测试结果终端输出")}
{_img_tag(screenshots.get("metrics_chart", Path("-")), "截图 3：各类别 AP50 与 mAP50 条形图")}

<h2>8. 评测产物图</h2>
{_img_tag(pr_curve, "PR 曲线（BoxPR_curve）")}
{_img_tag(cm, "归一化混淆矩阵")}
{_img_tag(pred_samples[0], "典型预测样例 val_batch0_pred")}
{_img_tag(pred_samples[1], "典型预测样例 val_batch1_pred")}
{_img_tag(pred_samples[2], "典型预测样例 val_batch2_pred")}

<h2>9. 测试执行溯源</h2>
<table>
<tr><th>项目</th><th>内容</th></tr>
<tr><td>git commit</td><td><code>{html.escape(git.get("commit", "-"))}</code></td></tr>
<tr><td>git 分支</td><td>{html.escape(git.get("branch", "-"))}</td></tr>
<tr><td>工作区状态</td><td>{"有未提交修改" if git.get("dirty") == "true" else "干净"}</td></tr>
<tr><td>主机名</td><td>{html.escape(env.get("hostname", "-"))}</td></tr>
<tr><td>操作系统</td><td>{html.escape(env.get("platform", "-"))}</td></tr>
<tr><td>Python</td><td>{html.escape(env.get("python_version", "-"))}</td></tr>
<tr><td>GPU</td><td><code>{html.escape(provenance.get("gpu", "-"))}</code></td></tr>
<tr><td>开始时间</td><td>{html.escape(provenance.get("started_at_iso", "-"))}</td></tr>
<tr><td>结束时间</td><td>{html.escape(provenance.get("finished_at_iso", "-"))}</td></tr>
<tr><td>总耗时</td><td>{provenance.get("duration_seconds", "-")} 秒</td></tr>
</table>

<h2>10. 结论</h2>
<p>本次第三方测试完成，主指标实测值为 <b>mAP@0.5 = {metrics["mAP50"]:.4f}</b>。
本报告仅记录实测结果；符合性判定以委托测试文件或测试机构正式规则为准。</p>

</body>
</html>
"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html_doc, encoding="utf-8")
    return save_path
