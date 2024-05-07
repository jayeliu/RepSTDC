# dataset settings
# custom_imports = dict(imports=['datasets.gid.GIDDataset'], allow_failed_imports=False)
dataset_type = 'OEMDataset'
data_root = 'data/OpenEarthMap'
img_norm_cfg = dict(
    mean=[116.79,119.38,103.96],
    std=[44.72, 40.65, 41.39],
    to_rgb=False)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile',imdecode_backend='tifffile'),
    dict(type='LoadAnnotations',reduce_zero_label=True,imdecode_backend='tifffile'),
    dict(
        type='RandomResize',
        scale=(2048,512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile',imdecode_backend='tifffile'),
    dict(type='Resize', scale=(2048,512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations',reduce_zero_label=True,imdecode_backend='tifffile'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile',imdecode_backend='tifffile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations',imdecode_backend='tifffile')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train', seg_map_path='labels/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val_1', seg_map_path='labels/val_1'),
        pipeline=test_pipeline))
test_dataloader =val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice','mFscore'])
test_evaluator = val_evaluator