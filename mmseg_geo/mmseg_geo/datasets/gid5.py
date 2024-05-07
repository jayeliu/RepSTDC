# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

# from .builder import DATASETS
# from .custom import CustomDataset


@DATASETS.register_module()
class GID5Dataset(BaseSegDataset):
    """ISPRS dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """

    METAINFO = dict(
        classes=(
            "unlabeled",
            "build-up",
            "farmland",
            "forest",
            "meadow",
            "water",
        ),
        palette=[
            [0,0,0],
            [255,0,0],
            [0,255,0],
            [0,255,255],
            [255,255,0],
            [0,0,255],
        ],
    )

    def __init__(self, **kwargs):
        super(GID5Dataset, self).__init__(
            img_suffix=".tif", seg_map_suffix="_5label.png", reduce_zero_label=False, **kwargs
        )
