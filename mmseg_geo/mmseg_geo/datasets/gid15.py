# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

# from .builder import DATASETS
# from .custom import CustomDataset


@DATASETS.register_module()
class GID15Dataset(BaseSegDataset):
    """ISPRS dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """

    METAINFO = dict(
        classes=(
         'unlabeled','industrial_land','urban_residential','rural_residential','traffic_land','paddy_field',
                          'irrigated_land','dry_cropland','garden_plot','arbor_woodland','shrub_land','natural_grassland',
                          'artificial_grassland','river','lake','pond'    
        ),
        palette=[
            [0,0,0],
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
        ],
    )

    def __init__(self, **kwargs):
        super(GID15Dataset, self).__init__(
            img_suffix=".tif", seg_map_suffix="_15label.png", reduce_zero_label=False, **kwargs
        )
