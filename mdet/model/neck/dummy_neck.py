import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mdet.model import BaseModule
from mdet.utils.factory import FI


@FI.register
class DummyNeck(BaseModule):
    def __init__(self):
        super().__init__()

    def forward_train(self, x):
        return x
