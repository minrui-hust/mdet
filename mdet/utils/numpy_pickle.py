import mdet.utils.shm_buffer as shm_buffer
from mdet.utils.shm_buffer import SHM_ROOT
import uuid
import numpy as np
from multiprocessing.reduction import ForkingPickler
from pytorch_lightning.utilities.distributed import rank_zero_only
import shutil
import os

r'''
reduction to pickle numpy.ndarray using shared memory,
which is bladely fast
'''


def rebuild_ndarray(shm_name: str):
    return shm_buffer.open(shm_name, manager=True).asarray().copy()


def reduce_ndarray(array: np.ndarray):
    shm_name = str(uuid.uuid4())
    tmp_array = shm_buffer.new(shm_name, array.shape,
                               array.dtype, manager=False).asarray()
    tmp_array[:] = array[:]
    return (rebuild_ndarray, (shm_name,))


@rank_zero_only
def init_shm_based_numpy_pickle():
    # prepare the shared memory dir
    shutil.rmtree(SHM_ROOT, ignore_errors=True)
    os.makedirs(SHM_ROOT, exist_ok=True)

    # register reduction
    ForkingPickler.register(np.ndarray, reduce_ndarray)


# call init
init_shm_based_numpy_pickle()
