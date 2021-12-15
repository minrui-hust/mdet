import pytorch_lightning as pl
from mdet.utils.factory import FI


class Runner(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # train
        self.train_model = FI.create(config['model']['train'])
        self.train_loader = FI.create(self.config['data']['train'])
        self.optimizer = FI.create(self.config['optimizer'])
        self.lr_scheduler = FI.create(self.config['lr_scheduler'])
        self.train_model.set_train()

        # eval(val, test)
        self.eval_model = FI.create(config['model']['eval'])
        self.val_loader = FI.create(self.config['data']['val'])
        self.test_loader = FI.create(self.config['data']['test'])
        self.eval_model.set_eval()

        # infer(export for depoly)
        self.infer_model = FI.create(config['model']['infer'])
        self.infer_model.set_infer()

        # eval and infer track the training (use same parameters)
        self.track_model(self.train_model, self.eval_model):
        self.track_model(self.train_model, self.infer_model):

    def forward(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self.train_model(batch)

    def training_epoch_end(self, epoch_output):
        pass

    def validation_step(self, batch, batch_idx):
        return self.eval_model(batch)

    def validation_epoch_end(self, epoch_output):
        pass

    def test_step(self, batch, batch_idx):
        return self.eval_model(batch)

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

    def track_model(self, m_target, m_follow):
        # track parameters
        for name in m_follow._parameters:
            m_follow._parameters[name] = m_target._parameters[name]

        # track buffers
        for name in m_follow._buffers:
            m_follow._buffers[name] = m_target._buffers[name]

        # recursively track submodules
        for name in m_follow._modules:
            self.track_model(
                m_target._modules[name], m_follow._modules[name])
