crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        90.76077249,
        95.9662333,
        87.58776207,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        59.24012407,
        58.03330621,
        57.22955588,
    ],
    type='SegDataPreProcessor')
data_root = 'data/gid15'
dataset_type = 'GID15Dataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmseg', by_epoch=False, interval=8000, type='CheckpointHook'),
    logger=dict(
        _scope_='mmseg',
        interval=50,
        log_metric_by_epoch=False,
        type='LoggerHook'),
    param_scheduler=dict(_scope_='mmseg', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmseg', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmseg', type='IterTimerHook'),
    visualization=dict(_scope_='mmseg', type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_norm_cfg = dict(
    mean=[
        90.76077249,
        95.9662333,
        87.58776207,
    ],
    std=[
        59.24012407,
        58.03330621,
        57.22955588,
    ],
    to_rgb=False)
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=[
        dict(
            align_corners=False,
            channels=64,
            concat_input=False,
            in_channels=128,
            in_index=2,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_classes=16,
            num_convs=1,
            sampler=dict(min_kept=10000, thresh=0.7, type='OHEMPixelSampler'),
            type='FCNHead'),
        dict(
            align_corners=False,
            channels=64,
            concat_input=False,
            in_channels=128,
            in_index=1,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_classes=16,
            num_convs=1,
            sampler=dict(min_kept=10000, thresh=0.7, type='OHEMPixelSampler'),
            type='FCNHead'),
        dict(
            align_corners=True,
            boundary_threshold=0.1,
            channels=64,
            concat_input=False,
            in_channels=256,
            in_index=0,
            loss_decode=[
                dict(
                    loss_name='loss_ce',
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=True),
                dict(loss_name='loss_dice', loss_weight=1.0, type='DiceLoss'),
            ],
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_classes=2,
            num_convs=1,
            type='STDCHead'),
    ],
    backbone=dict(
        backbone_cfg=dict(
            act_cfg=dict(type='ReLU'),
            bottleneck_type='cat',
            channels=(
                32,
                64,
                256,
                512,
                1024,
            ),
            in_channels=3,
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_convs=4,
            stdc_type='STDCNet1',
            type='RepSTDCNet',
            with_final_conv=False),
        ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4),
        fusion_type='CA',
        last_in_channels=(
            1024,
            512,
        ),
        out_channels=128,
        type='RepSTDCContextPathNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            90.76077249,
            95.9662333,
            87.58776207,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            59.24012407,
            58.03330621,
            57.22955588,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=True,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=256,
        in_index=3,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=16,
        num_convs=1,
        sampler=dict(min_kept=10000, thresh=0.7, type='OHEMPixelSampler'),
        type='FCNHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    _scope_='mmseg',
    clip_grad=None,
    loss_scale='dynamic',
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='AmpOptimWrapper')
optimizer = dict(
    _scope_='mmseg', lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=0.1, type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=80000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(_scope_='mmseg', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='imgs/val', seg_map_path='gts/val'),
        data_root='data/gid15',
        pipeline=[
            dict(imdecode_backend='tifffile', type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='GID15Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(imdecode_backend='tifffile', type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    _scope_='mmseg',
    max_iters=80000,
    type='IterBasedTrainLoop',
    val_interval=8000)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_prefix=dict(img_path='imgs/train', seg_map_path='gts/train'),
        data_root='data/gid15',
        pipeline=[
            dict(imdecode_backend='tifffile', type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    1.0,
                    2.0,
                ),
                scale=(
                    512,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='GID15Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(imdecode_backend='tifffile', type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            1.0,
            2.0,
        ),
        scale=(
            512,
            512,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(
        backend_args=None,
        imdecode_backend='tifffile',
        type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(_scope_='mmseg', type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='imgs/val', seg_map_path='gts/val'),
        data_root='data/gid15',
        pipeline=[
            dict(imdecode_backend='tifffile', type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='GID15Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = './work_dirs/repstdc-ca_512x512_80k_gid15'
