import numpy as np
from tqdm import tqdm

from mdet.utils.factory import FI
import mdet.utils.io as io


@FI.register
class ArgoPredSummary(object):
    def __init__(self):
        super().__init__()

    def __call__(self, root_path, split):
        summary(root_path, split)


def summary(root_path, split):
    pass


def summary_map(root_path):
    pass


def summary_agent(root_path, split):
    pass
