PROJECT_ROOT = "/home/bm/Dev/Microlens_DF"
DATA_ROOT = f"{PROJECT_ROOT}/output/experiments/phase3_segmentation/private_weak_masks"
work_dir = f"{PROJECT_ROOT}/output/experiments/phase3_segmentation/batch2_hrnet_ocr_w18_private"

default_scope = "mmseg"

crop_size = (512, 512)
num_classes = 4
metainfo = dict(
    classes=("background", "scratch", "spot", "damage"),
    palette=[[0, 0, 0], [0, 220, 80], [80, 220, 255], [255, 80, 80]],
)

data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[127.5],
    std=[127.5],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

train_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="LoadAnnotations"),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.98),
    dict(type="PackSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="BaseSegDataset",
        data_root=DATA_ROOT,
        ann_file="splits/segmentation_train.txt",
        data_prefix=dict(img_path="images", seg_map_path="masks"),
        img_suffix=".png",
        seg_map_suffix=".png",
        reduce_zero_label=False,
        metainfo=metainfo,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="BaseSegDataset",
        data_root=DATA_ROOT,
        ann_file="splits/segmentation_val.txt",
        data_prefix=dict(img_path="images", seg_map_path="masks"),
        img_suffix=".png",
        seg_map_suffix=".png",
        reduce_zero_label=False,
        metainfo=metainfo,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader
val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice"])
test_evaluator = val_evaluator

norm_cfg = dict(type="BN", requires_grad=True)
hrnet_extra = dict(
    stage1=dict(num_modules=1, num_branches=1, block="BOTTLENECK", num_blocks=(4,), num_channels=(64,)),
    stage2=dict(num_modules=1, num_branches=2, block="BASIC", num_blocks=(4, 4), num_channels=(18, 36)),
    stage3=dict(num_modules=4, num_branches=3, block="BASIC", num_blocks=(4, 4, 4), num_channels=(18, 36, 72)),
    stage4=dict(
        num_modules=3,
        num_branches=4,
        block="BASIC",
        num_blocks=(4, 4, 4, 4),
        num_channels=(18, 36, 72, 144),
    ),
)

model = dict(
    type="CascadeEncoderDecoder",
    num_stages=2,
    data_preprocessor=data_preprocessor,
    backbone=dict(type="HRNet", in_channels=1, norm_cfg=norm_cfg, norm_eval=False, extra=hrnet_extra),
    decode_head=[
        dict(
            type="FCNHead",
            in_channels=[18, 36, 72, 144],
            in_index=(0, 1, 2, 3),
            input_transform="resize_concat",
            channels=270,
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
        ),
        dict(
            type="OCRHead",
            in_channels=[18, 36, 72, 144],
            in_index=(0, 1, 2, 3),
            input_transform="resize_concat",
            channels=512,
            ocr_channels=256,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        ),
    ],
    train_cfg=dict(),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(384, 384)),
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(type="AdamW", lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-4, by_epoch=True, begin=0, end=3),
    dict(type="CosineAnnealingLR", T_max=50, by_epoch=True, begin=3, end=50, eta_min=1e-6),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=4)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=10, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=4,
        save_best="mIoU",
        rule="greater",
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
