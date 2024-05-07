# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

# from .builder import DATASETS
# from .custom import CustomDataset


@DATASETS.register_module()
class FBP24Dataset(BaseSegDataset):
    """ISPRS dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """

    METAINFO = dict(
        classes=(
            "unlabeled",
            "industrial area",
            "paddy field",
            "irrigated field",
            "dry cropland",
            "garden land",
            "arbor forest",
            "shrub forest",
            "park",
            "natural meadow",
            "artificial meadow",
            "river",
            "urban residential",
            "lake",
            "pond",
            "fish pond",
            "snow",
            "bareland",
            "rural residential",
            "stadium",
            "square",
            "road",
            "overpass",
            "railway station",
            "airport",
        ),
        palette=[
            [0,0,0],
            [200, 0, 0],
            [0, 200, 0],
            [150, 250, 0],
            [150, 200, 150],
            [200, 0, 200],
            [150, 0, 250],
            [150, 150, 250],
            [200, 150, 200],
            [250, 200, 0],
            [200, 200, 0],
            [0, 0, 200],
            [250, 0, 150],
            [0, 150, 200],
            [0, 200, 250],
            [150, 200, 250],
            [250, 250, 250],
            [200, 200, 200],
            [200, 150, 150],
            [250, 200, 150],
            [150, 150, 0],
            [250, 150, 150],
            [250, 150, 0],
            [250, 200, 250],
            [200, 150, 0],
        ],
    )

    def __init__(self, **kwargs):
        super(FBP24Dataset, self).__init__(
            img_suffix=".tif", seg_map_suffix="_24label.png", reduce_zero_label=False, **kwargs
        )
