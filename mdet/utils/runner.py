import pytorch_lightning as pl
from mdet.utils.factory import FI


class Runner(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # train
        self.train_model = FI.create(config['model']['train'])
        self.loss = FI.create(config['loss'])
        self.optimizer = FI.create(self.config['optimizer'])
        self.lr_scheduler = FI.create(self.config['lr_scheduler'])
        self.train_loader = FI.create(self.config['data']['train'])

        # eval(val, test)
        self.eval_model = FI.create(config['model']['eval'])
        self.eval_output = FI.create(config['output']['eval'])
        self.val_loader = FI.create(self.config['data']['val'])
        self.test_loader = FI.create(self.config['data']['test'])

        # infer
        self.infer_model = FI.create(config['model']['infer'])
        self.ifner_output = FI.create(config['output']['infer'])

        # set model state
        self.train_model.set_train()
        self.eval_model.set_eval()
        self.infer_model.set_infer()

        # eval and infer track the parameters of training
        for p_train, p_eval, p_infer in zip(self.train_model.parameters(), self.eval_model.parameters(), self.infer_model.parameters()):
            p_eval = p_train
            p_infer = p_train

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        out = self.train_model(batch)
        loss = self.loss(out, batch)
        return loss  # loss is a single tensor or a dict, if it is a dict, it must has key 'loss'

    def training_epoch_end(self, epoch_output):
        pass

    def validation_step(self, batch, batch_idx):
        out = self.eval_model(batch)
        result = self.eval_output(out, batch)
        return result

    def validation_epoch_end(self, epoch_output):
        pass

    def test_step(self, batch, batch_idx):
        out = self.eval_model(batch)
        result = self.eval_output(out, batch)
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
