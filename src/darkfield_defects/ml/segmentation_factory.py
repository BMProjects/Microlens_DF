"""分割模型工厂与 checkpoint 元数据解析."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch.nn as nn
import torch.nn.functional as F

from darkfield_defects.ml.hrnet_ocr import HRNetOCRW18
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
    hf_model_id: str | None = None

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


class HfSegformerWrapper(nn.Module):
    """Hugging Face SegFormer 语义分割封装.

    当前默认实现为 SegFormer-B2 结构参数，并统一输出与输入同尺度 logits，
    方便复用现有训练/推理代码。
    """

    def __init__(self, spec: SegmentationModelSpec):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "需要 transformers 才能构建 SegFormer: pip install darkfield-defects[hfseg]"
            ) from exc

        model_id = spec.hf_model_id or ""
        if model_id:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=spec.num_classes,
                num_channels=spec.in_channels,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig(
                num_labels=spec.num_classes,
                num_channels=spec.in_channels,
                depths=[3, 4, 6, 3],
                hidden_sizes=[64, 128, 320, 512],
                decoder_hidden_size=256,
                num_attention_heads=[1, 2, 5, 8],
                sr_ratios=[8, 4, 2, 1],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                classifier_dropout_prob=0.1,
            )
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def build_segmentation_model(spec: SegmentationModelSpec):
    """根据 spec 构建分割模型."""
    model_name = spec.model_name.lower()
    if model_name == "light_unet":
        return LightUNet(
            in_channels=spec.in_channels,
            num_classes=spec.num_classes,
            base_features=spec.base_features,
        )

    if model_name in {"hrnet_ocr_w18", "hrnet-ocr-w18", "hrnetocrw18"}:
        return HRNetOCRW18(
            in_channels=spec.in_channels,
            num_classes=spec.num_classes,
        )

    if model_name in {"segformer_b2_hf", "segformer_hf"}:
        return HfSegformerWrapper(spec)

    if model_name not in SMP_MODEL_ALIASES:
        supported = ", ".join(["light_unet", "segformer_b2_hf", "hrnet_ocr_w18", *sorted(SMP_MODEL_ALIASES)])
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
        hf_model_id=ckpt.get("hf_model_id"),
    )
