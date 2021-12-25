import torch
from mdet.model import BaseModule


class BasePostProcess(BaseModule):
    def __init__(self):
        super().__init__()

    def loss(self, result, batch):
        raise NotImplementedError

    @torch.no_grad()
    def eval(self, result):
        raise NotImplementedError

    @torch.no_grad()
    def infer(self, result):
        self.eval(result)
