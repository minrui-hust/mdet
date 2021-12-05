from torch.utils.data.dataloader import DataLoader
from mdet.utils.factory import FI


@FI.register
class BaseDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
