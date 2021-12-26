from multiprocessing import Process, Queue
import time
import random
import os
import numpy as np
from tqdm import tqdm
from multiprocessing.reduction import ForkingPickler

import mdet.utils.shm_buffer as shm_buffer
import uuid
from mdet.utils.shm_buffer.ism_buffer import lib


def rebuild_ndarray(shm_name: str):
    return shm_buffer.open(shm_name, manager=True).asarray()


def reduce_ndarray(array: np.ndarray):
    shm_name = str(uuid.uuid4())
    tmp_array = shm_buffer.new(shm_name, array.shape,
                               array.dtype, manager=False).asarray()
    tmp_array[:] = array[:]
    return (rebuild_ndarray, (shm_name,))


def consumer(q: Queue):
    for _ in tqdm(range(1000000)):
        a = q.get()
        del a


def producer(q: Queue):
    for _ in range(1000000):
        a = np.random.rand(1024*1024)
        q.put(a)


if __name__ == '__main__':
    ForkingPickler.register(np.ndarray, reduce_ndarray)

    q = Queue(maxsize=4)
    p1 = Process(target=producer, args=(q,))
    c1 = Process(target=consumer, args=(q,))

    # 开始
    p1.start()
    c1.start()
    print('Started')

    p1.join()
    c1.join()
    print('Done')
