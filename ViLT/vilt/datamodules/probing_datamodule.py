from vilt.datasets import ProbingDataset
from .datamodule_base import BaseDataModule


class ProbingDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ProbingDataset

    @property
    def dataset_cls_no_false(self):
        return ProbingDataset

    @property
    def dataset_name(self):
        return "probing"
