"""分割模型工厂与 checkpoint 元数据解析."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from darkfield_defects.ml.models import LightUNet


@dataclass(slots=True)
class SegmentationModelSpec:
    """统一描述分割模型的结构参数."""

    model_name: str = "light_unet"
    in_channels: int = 1
    num_classes: int = 4
    base_features: int = 64
    encoder_name: str | None = None
    encoder_weights: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


SMP_MODEL_ALIASES = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "deeplabv3": "DeepLabV3",
    "deeplabv3plus": "DeepLabV3Plus",
    "fpn": "FPN",
    "pspnet": "PSPNet",
    "linknet": "Linknet",
    "manet": "MAnet",
    "pan": "PAN",
}


def build_segmentation_model(spec: SegmentationModelSpec):
    """根据 spec 构建分割模型."""
    model_name = spec.model_name.lower()
    if model_name == "light_unet":
        return LightUNet(
            in_channels=spec.in_channels,
            num_classes=spec.num_classes,
            base_features=spec.base_features,
        )

    if model_name not in SMP_MODEL_ALIASES:
        supported = ", ".join(["light_unet", *sorted(SMP_MODEL_ALIASES)])
        raise ValueError(f"不支持的分割模型: {spec.model_name}. 可选: {supported}")

    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "需要 segmentation-models-pytorch 才能构建该分割模型: "
            "pip install darkfield-defects[ml]"
        ) from exc

    smp_cls = getattr(smp, SMP_MODEL_ALIASES[model_name])
    encoder_name = spec.encoder_name or "resnet34"
    return smp_cls(
        encoder_name=encoder_name,
        encoder_weights=spec.encoder_weights,
        in_channels=spec.in_channels,
        classes=spec.num_classes,
    )


def spec_from_checkpoint(ckpt: dict[str, Any]) -> SegmentationModelSpec:
    """从 checkpoint 中恢复模型结构信息."""
    return SegmentationModelSpec(
        model_name=ckpt.get("model_name", "light_unet"),
        in_channels=ckpt.get("in_channels", 1),
        num_classes=ckpt.get("num_classes", 4),
        base_features=ckpt.get("base_features", 64),
        encoder_name=ckpt.get("encoder_name"),
        encoder_weights=ckpt.get("encoder_weights"),
    )
