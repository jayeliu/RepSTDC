# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

# from .builder import DATASETS
# from .custom import CustomDataset


@DATASETS.register_module()
class GIDDataset(BaseSegDataset):
    """ISPRS dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """

    METAINFO = dict(
        classes=(
            "Industrial_Land",
            "Urban_Residential",
            "Rural_Residential",
            "Traffic_Land",
            "Paddy_Field",
            "Irrigated_Land",
            "Dry_Cropland",
            "Garden_Plot",
            "Arbor_Woodland",
            "Shrub_Land",
            "Natural_Grassland",
            "Artifical_Grassland",
            "River",
            "Lake",
            "Pond",
            "unlabeled"
        ),
        palette=[
            [200, 0, 0],
            [250, 0, 150],
            [200, 150, 150],
            [250, 150, 150],
            [0, 200, 0],
            [150, 250, 0],
            [150, 200, 150],
            [200, 0, 200],
            [150, 0, 250],
            [150, 150, 250],
            [250, 200, 0],
            [200, 200, 0],
            [0, 0, 200],
            [0, 150, 200],
            [0, 200, 250],
            [0,0,0]
        ],
    )

    def __init__(self, **kwargs):
        super(GIDDataset, self).__init__(
            img_suffix=".tif", seg_map_suffix=".tif", reduce_zero_label=False, **kwargs
        )
