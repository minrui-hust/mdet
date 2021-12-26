from multiprocessing import Process, Queue
import time
import random
import os
import numpy as np
from tqdm import tqdm


def consumer(q):
    for _ in tqdm(range(100000000)):
        res = q.get()
        del res


def producer(q):
    while True:
        a = np.random.rand(1024*1024)
        q.put(a)


if __name__ == '__main__':
    q = Queue()
    p1 = Process(target=producer, args=(q,))
    c1 = Process(target=consumer, args=(q,))

    # 开始
    p1.start()
    c1.start()
    print('Started')
