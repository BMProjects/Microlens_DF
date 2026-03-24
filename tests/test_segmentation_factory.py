from __future__ import annotations

import pytest

from darkfield_defects.ml.models import LightUNet
from darkfield_defects.ml.segmentation_factory import (
    SegmentationModelSpec,
    build_segmentation_model,
    spec_from_checkpoint,
)


def test_build_light_unet_from_spec() -> None:
    model = build_segmentation_model(
        SegmentationModelSpec(model_name="light_unet", in_channels=1, num_classes=4)
    )
    assert isinstance(model, LightUNet)


def test_spec_from_checkpoint_defaults() -> None:
    spec = spec_from_checkpoint({"model_state_dict": {}})
    assert spec.model_name == "light_unet"
    assert spec.in_channels == 1
    assert spec.num_classes == 4


def test_build_segmentation_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        build_segmentation_model(SegmentationModelSpec(model_name="unknown_model"))
