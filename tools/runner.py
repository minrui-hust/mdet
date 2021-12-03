import pytorch_lightning as pl
from factory.component_factory import ComponentFactory

CF = ComponentFactory()


class Runner(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model = CF.createComponent(config['model'])
        self.loss = CF.createComponent(config['loss'])
        self.post_process = CF.createComponent(config['post_process'])

        self.train_dataloader = CF.createComponent(self.config['train_loader'])
        self.val_dataloader = CF.createComponent(self.config['val_loader'])
        self.test_dataloader = CF.createComponent(self.config['test_loader'])

        self.optimizer = CF.createComponent(self.config['optimizer'])
        self.lr_scheduler = CF.createComponent(self.config['lr_scheduler'])

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        out = self.model(batch)
        loss = self.loss(out, batch)
        return loss  # loss is a single tensor or a dict, if it is a dict, it must has key 'loss'

    def training_epoch_end(self, epoch_output):
        pass

    def validation_step(self, batch, batch_idx):
        out = self.model(batch)
        result = self.post_process(out, batch)
        return result

    def validation_epoch_end(self, epoch_output):
        pass

    def test_step(self, batch, batch_idx):
        out = self.model(batch)
        result = self.post_process(out, batch)
        return result

    def test_epoch_end(self, epoch_output):
        pass

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader

    def test_dataloader(self):
        return self.test_dataloader

    def configure_optimizers(self):
        return [self.optimizer], [self.lr_scheduler]
