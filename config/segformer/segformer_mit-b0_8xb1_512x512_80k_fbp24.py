_base_ = [
    "mmseg::_base_/models/segformer_mit-b0.py",
    "../_base_/datasets/fbp24.py",
    '../_base_/default_runtime.py', 'mmseg::_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size, mean=[126.72574477,  90.76077249,  95.9662333 ,  87.58776207],
    std=[62.913678  , 59.24012407, 58.03330621, 57.22955588])
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        in_channels=4,
      ),
    decode_head=dict(
        num_classes=25,
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=80000,
        by_epoch=False,
    )
]
train_dataloader = dict(batch_size=32, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader