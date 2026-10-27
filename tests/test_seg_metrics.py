import numpy as np
import pytest

from darkfield_defects.eval.seg_metrics import ConfusionMatrix, all_background_pa


def test_fcn_metrics_hand_computed():
    gt = np.array([[0, 0, 1, 1], [0, 2, 2, 255]])
    pred = np.array([[0, 1, 1, 1], [0, 2, 0, 0]])
    cm = ConfusionMatrix(3)
    cm.update(gt, pred)
    m = cm.metrics(["bg", "a", "b"])
    # 计入 7 个像素（255 不计）；正确 5 个
    assert m["pixels"] == 7
    assert m["PA"] == pytest.approx(5 / 7)
    # 各类：bg t=3 tp=2 p=3；a t=2 tp=2 p=3；b t=2 tp=1 p=1
    iou = [2 / 4, 2 / 3, 1 / 2]
    assert m["mIoU"] == pytest.approx(np.mean(iou))
    assert m["MPA"] == pytest.approx(np.mean([2 / 3, 1.0, 1 / 2]))
    assert m["FWIoU"] == pytest.approx((3 * iou[0] + 2 * iou[1] + 2 * iou[2]) / 7)
    assert m["per_class"]["b"]["Dice"] == pytest.approx(2 * 1 / (2 + 1), abs=1e-5)   # 分类别值保留 5 位小数
    assert all_background_pa(cm) == pytest.approx(3 / 7)


def test_absent_class_is_skipped_and_accumulates():
    cm = ConfusionMatrix(3)
    cm.update(np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8))
    cm.update(np.ones((1, 2), np.uint8), np.ones((1, 2), np.uint8))
    m = cm.metrics()
    assert m["PA"] == 1.0 and m["mIoU"] == 1.0          # 类别 2 未出现：NaN 跳过
    assert m["per_class"]["2"]["IoU"] is None


def test_out_of_range_rejected():
    with pytest.raises(ValueError):
        ConfusionMatrix(2).update(np.array([[2]]), np.array([[0]]))
