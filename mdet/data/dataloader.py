from torch.utils.data.dataloader import DataLoader as RawDataLoader
from mdet.utils.factory import FI


@FI.register
class Dataloader(RawDataLoader):
    def __init__(self, dataset, collator, **kwargs):
        super().__init__(dataset=FI.create(dataset),
                         collate_fn=FI.create(collator), **kwargs)
