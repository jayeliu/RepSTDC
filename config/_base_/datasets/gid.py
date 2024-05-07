# dataset settings
# custom_imports = dict(imports=['datasets.gid.GIDDataset'], allow_failed_imports=False)
dataset_type = 'GIDDataset'
data_root = 'data/GID'
img_norm_cfg = dict(
    mean=[126.72574477,  90.76077249,  95.9662333 ,  87.58776207],
    std=[62.913678  , 59.24012407, 58.03330621, 57.22955588],
    to_rgb=False)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile',imdecode_backend='tifffile',),
    dict(type='LoadAnnotations',imdecode_backend='tifffile'),
    dict(
        type='RandomResize',
        scale=(512,512),
        ratio_range=(1.0, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile',imdecode_backend='tifffile'),
    dict(type='Resize', scale=(512,512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations',imdecode_backend='tifffile'),
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
    batch_size=2,
    num_workers=2,
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
            img_path='images/val', seg_map_path='labels/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
# train_pipeline = [
#     dict(type='LoadImageFromFile',imdecode_backend='tifffile',to_float32=True),
#     dict(type='LoadAnnotations', reduce_zero_label=True,imdecode_backend='tifffile'),
#     dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     # dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile',imdecode_backend='tifffile',to_float32=True),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1024, 1024),
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(
#     samples_per_gpu=24,
#     workers_per_gpu=4,
#     train=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='images/train',
#         ann_dir='masks/train',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='images/val',
#         ann_dir='masks/val',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='images/val',
#         ann_dir='masks/val',
#         pipeline=test_pipeline))
