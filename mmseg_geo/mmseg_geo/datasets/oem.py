# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

# from .builder import DATASETS
# from .custom import CustomDataset


@DATASETS.register_module()
class OEMDataset(BaseSegDataset):
    """OpenEarthMap dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """

    METAINFO = dict(
        classes=(
            "Bareland",
            "Rangeland",
            "Developed Space",
            "Road",
            "Tree",
            "Water",
            "Agriculture land",
            "Building"
        ),
        palette=[
            [128,0,0],
            [0, 255, 36],
            [148, 148, 148],
            [255, 255, 255],
            [34, 97, 38],
            [0, 69, 255],
            [75, 181, 73],
            [222, 31, 7]
        ],
    )

    def __init__(self, **kwargs):
        super(OEMDataset, self).__init__(
            img_suffix=".tif", seg_map_suffix=".tif", reduce_zero_label=True, **kwargs
        )
