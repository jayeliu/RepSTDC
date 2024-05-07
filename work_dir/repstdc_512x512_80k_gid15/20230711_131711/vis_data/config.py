norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[90.76077249, 95.9662333, 87.58776207],
    std=[59.24012407, 58.03330621, 57.22955588],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[90.76077249, 95.9662333, 87.58776207],
        std=[59.24012407, 58.03330621, 57.22955588],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)),
    pretrained=None,
    backbone=dict(
        type='RepSTDCContextPathNet',
        backbone_cfg=dict(
            type='RepSTDCNet',
            stdc_type='STDCNet1',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='cat',
            num_convs=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            with_final_conv=False),
        last_in_channels=(1024, 512),
        out_channels=128,
        ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4)),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        channels=256,
        num_convs=1,
        num_classes=16,
        in_index=3,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=True,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=16,
            in_index=2,
            norm_cfg=dict(type='BN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=16,
            in_index=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='STDCHead',
            in_channels=256,
            channels=64,
            num_convs=1,
            num_classes=2,
            boundary_threshold=0.1,
            in_index=0,
            norm_cfg=dict(type='BN', requires_grad=True),
            concat_input=False,
            align_corners=True,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    loss_name='loss_ce',
                    use_sigmoid=True,
                    loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
            ])
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'GID15Dataset'
data_root = 'data/gid15'
img_norm_cfg = dict(
    mean=[90.76077249, 95.9662333, 87.58776207],
    std=[59.24012407, 58.03330621, 57.22955588],
    to_rgb=False)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(1.0, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(
        type='LoadImageFromFile',
        imdecode_backend='tifffile',
        backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale_factor': 0.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 0.75,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.0,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.25,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.75,
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'prob': 0.0,
                        'direction': 'horizontal'
                    }, {
                        'type': 'RandomFlip',
                        'prob': 1.0,
                        'direction': 'horizontal'
                    }], [{
                        'type': 'LoadAnnotations'
                    }], [{
                        'type': 'PackSegInputs'
                    }]])
]
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='GID15Dataset',
        data_root='data/gid15',
        data_prefix=dict(img_path='imgs/train', seg_map_path='gts/train'),
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
            dict(type='LoadAnnotations'),
            dict(
                type='RandomResize',
                scale=(512, 512),
                ratio_range=(1.0, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='GID15Dataset',
        data_root='data/gid15',
        data_prefix=dict(img_path='imgs/val', seg_map_path='gts/val'),
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='GID15Dataset',
        data_root='data/gid15',
        data_prefix=dict(img_path='imgs/val', seg_map_path='gts/val'),
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
val_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, _scope_='mmseg')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None,
    _scope_='mmseg')
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,
        begin=1000,
        end=80000,
        by_epoch=False)
]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=80000,
    val_interval=8000,
    _scope_='mmseg')
val_cfg = dict(type='ValLoop', _scope_='mmseg')
test_cfg = dict(type='TestLoop', _scope_='mmseg')
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmseg'),
    logger=dict(
        type='LoggerHook',
        interval=50,
        log_metric_by_epoch=False,
        _scope_='mmseg'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmseg'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=8000, _scope_='mmseg'),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmseg'),
    visualization=dict(type='SegVisualizationHook', _scope_='mmseg'))
launcher = 'none'
work_dir = './work_dirs/repstdc_512x512_80k_gid15'
