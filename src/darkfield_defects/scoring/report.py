"""报告生成模块 — JSON + HTML 磨损评估报告."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from darkfield_defects.logging import get_logger
from darkfield_defects.scoring.quantify import WearMetrics
from darkfield_defects.scoring.wear_score import WearAssessment

logger = get_logger(__name__)


def _format_length_mm(value_mm: float) -> str:
    return f"{value_mm:.3f} mm"


def _format_area_mm2(value_mm2: float) -> str:
    return f"{value_mm2:.4f} mm²"


class _NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码器."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_json_report(
    filename: str,
    metrics: WearMetrics,
    assessment: WearAssessment,
    output_path: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """生成 JSON 格式的磨损评估报告."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "source_image": filename,
        "metrics": metrics.to_dict(),
        "assessment": assessment.to_dict(),
    }
    if extra:
        report["extra"] = extra

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    logger.info(f"JSON 报告已生成: {path}")
    return path


def generate_html_report(
    filename: str,
    metrics: WearMetrics,
    assessment: WearAssessment,
    output_path: str | Path,
    overlay_path: str | None = None,
) -> Path:
    """生成 HTML 格式的可视化磨损评估报告."""
    grade_colors = {"A": "#22c55e", "B": "#3b82f6", "C": "#f59e0b", "D": "#ef4444"}
    grade_color = grade_colors.get(assessment.grade, "#888")

    overlay_html = ""
    if overlay_path:
        overlay_html = f'<img src="{overlay_path}" style="max-width:100%; border-radius:8px;" />'

    # 贡献因素条形图
    contrib_bars = ""
    if assessment.contributors:
        max_val = max(assessment.contributors.values()) if assessment.contributors.values() else 1

        factor_cn = {
            "center_length": "中心区长度",
            "effective_length": "区域加权长度",
            "defect_count": "缺陷数量",
            "defect_area": "缺陷面积",
            "scatter_intensity": "散射强度",
        }

        for key, val in sorted(assessment.contributors.items(), key=lambda x: -x[1]):
            pct = val / max(max_val, 0.01) * 100
            label = factor_cn.get(key, key)
            contrib_bars += f"""
            <div style="margin:4px 0;">
              <span style="display:inline-block;width:100px;font-size:13px;">{label}</span>
              <div style="display:inline-block;width:200px;background:#333;border-radius:4px;overflow:hidden;vertical-align:middle;">
                <div style="width:{pct:.0f}%;height:16px;background:{grade_color};"></div>
              </div>
              <span style="font-size:12px;color:#aaa;margin-left:6px;">{val:.1f}</span>
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>镜片磨损评估报告 — {filename}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }}
  .container {{ max-width: 1000px; margin: 0 auto; }}
  .header {{ text-align: center; padding: 20px 0; }}
  .header h1 {{ color: #fff; font-size: 24px; margin: 0; }}
  .header .subtitle {{ color: #888; font-size: 14px; }}
  .grade-badge {{
    display: inline-block; font-size: 48px; font-weight: bold;
    color: {grade_color}; border: 3px solid {grade_color};
    border-radius: 50%; width: 80px; height: 80px;
    line-height: 80px; text-align: center; margin: 16px 0;
  }}
  .score {{ font-size: 28px; color: {grade_color}; font-weight: bold; }}
  .card {{ background: #16213e; border-radius: 12px; padding: 20px; margin: 16px 0; }}
  .card h2 {{ color: #fff; font-size: 18px; margin-top: 0; border-bottom: 1px solid #333; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  td, th {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #2a2a4a; }}
  th {{ color: #aaa; font-weight: normal; width: 40%; }}
  .conclusion {{ font-size: 15px; line-height: 1.6; color: #ccc; }}
  .zone-center {{ color: #ef4444; }}
  .zone-microstructure {{ color: #f59e0b; }}
  .zone-edge {{ color: #22c55e; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🔬 镜片磨损评估报告</h1>
    <div class="subtitle">{filename} | {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
  </div>

  <div class="card" style="text-align:center;">
    <div class="grade-badge">{assessment.grade}</div>
    <div class="score">{assessment.score:.1f} / 100</div>
    <div style="color:#aaa;margin-top:8px;">{assessment.grade_label}</div>
  </div>

  {f'<div class="card">{overlay_html}</div>' if overlay_html else ''}

  <div class="card">
    <h2>📊 磨损指标</h2>
    <table>
      <tr><th>总划痕数</th><td>{metrics.N_total}</td></tr>
      <tr><th>总骨架长度</th><td>{_format_length_mm(metrics.L_total * metrics.pixel_size_mm)}</td></tr>
      <tr><th>总划痕面积</th><td>{_format_area_mm2(metrics.A_total * metrics.pixel_size_mm ** 2)}</td></tr>
      <tr><th>长度密度</th><td>{metrics.D_density / max(metrics.pixel_size_mm, 1e-12):.4f} mm/mm²</td></tr>
      <tr><th class="zone-center">中心区划痕数</th><td>{metrics.N_center}</td></tr>
      <tr><th class="zone-center">中心区长度</th><td>{_format_length_mm(metrics.L_center * metrics.pixel_size_mm)}</td></tr>
      <tr><th class="zone-microstructure">微结构区长度</th><td>{_format_length_mm(metrics.L_microstructure * metrics.pixel_size_mm)}</td></tr>
      <tr><th class="zone-edge">边缘区长度</th><td>{_format_length_mm(metrics.L_edge * metrics.pixel_size_mm)}</td></tr>
      <tr><th>严重缺陷数</th><td>{metrics.N_critical}</td></tr>
      <tr><th>微结构区密度</th><td>{metrics.D_micro_density / max(metrics.pixel_size_mm, 1e-12):.4f} mm/mm²</td></tr>
      <tr><th>散射强度</th><td>{metrics.S_scatter:.2f}</td></tr>
      <tr><th>交叉点数量</th><td>{metrics.n_crossings}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>⚡ 场景风险</h2>
    <table>
      <tr><th>眩光指数 (Night Glare)</th><td>{assessment.glare_index:.1f}</td></tr>
      <tr><th>雾化指数 (Haze)</th><td>{assessment.haze_index:.1f}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>📈 贡献因素</h2>
    {contrib_bars}
  </div>

  <div class="card">
    <h2>💡 评估结论</h2>
    <div class="conclusion">{assessment.conclusion}</div>
  </div>
</div>
</body>
</html>"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML 报告已生成: {path}")
    return path
