_base_=['./repstdc_512x512_80k_oem.py']
crop_size = (512,512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor,
             backbone=dict(
                 fusion_type="CA",
             )
             )