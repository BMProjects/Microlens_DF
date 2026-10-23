"""单图像判定 δᵢ 评估测试."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from darkfield_defects.eval.image_level import (
    evaluate_image_level,
    parse_tile_name,
    read_yolo_txt,
    tile_box_f1,
)


def test_parse_tile_name_keeps_dashes_in_lens_id():
    assert parse_tile_name("37r-ljj_y0480_x1120") == ("37r-ljj", 480, 1120)
    with pytest.raises(ValueError):
        parse_tile_name("bad_name")


def test_tile_box_f1_is_class_aware():
    g = [(0, 0.5, 0.5, 0.2, 0.2, 1.0)]
    assert tile_box_f1(g, [(0, 0.5, 0.5, 0.2, 0.2, 0.9)]) == 1.0
    assert tile_box_f1(g, [(1, 0.5, 0.5, 0.2, 0.2, 0.9)]) == 0.0
    assert tile_box_f1([], []) == 1.0
    # 1 TP + 1 FP → F1 = 2/3
    assert tile_box_f1(g, [(0, 0.5, 0.5, 0.2, 0.2, 0.9), (0, 0.1, 0.1, 0.05, 0.05, 0.8)]) \
        == pytest.approx(2 / 3)


def test_read_yolo_txt_filters_conf_and_missing_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.1 0.1 0.9\n1 0.2 0.2 0.1 0.1 0.1\n")
    assert len(read_yolo_txt(p, conf_min=0.2)) == 1
    assert read_yolo_txt(tmp_path / "missing.txt") == []


def test_identical_predictions_give_full_accuracy(tmp_path):
    gt_dir, pred_dir, img_dir = (tmp_path / n for n in ("gt", "pred", "img"))
    for d in (gt_dir, pred_dir, img_dir):
        d.mkdir()
    img = np.full((1280, 1280), 20, dtype=np.uint8)
    cv2.line(img, (500, 600), (700, 640), 200, 3)
    cv2.imwrite(str(img_dir / "L1.png"), img)
    stems = ["L1_y0000_x0000", "L1_y0480_x0480"]
    lines = {stems[0]: "0 0.85 0.95 0.3 0.06\n", stems[1]: "1 0.5 0.5 0.05 0.05\n"}
    for s, txt in lines.items():
        (gt_dir / f"{s}.txt").write_text(txt)
        (pred_dir / f"{s}.txt").write_text(txt.replace("\n", " 0.9\n"))

    rep = evaluate_image_level(stems, gt_dir, pred_dir, img_dir, conf=0.2)
    s = rep.summary()
    assert s["lens_grade_accuracy"] == 1.0 and s["n_lenses"] == 1
    assert s["tile_f1@iou0.3>=0.5_accuracy"] == 1.0 and s["n_tiles"] == 2
    assert rep.lens_rows[0]["gt_score"] == rep.lens_rows[0]["pred_score"]
