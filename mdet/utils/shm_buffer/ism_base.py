
import numpy
import json


class ISMBase:
    _MAGIC_COOKIE = b'\xF0\x0A'

    @classmethod
    def open(cls, name, manager=False):
        """Open an existing shared interprocess numpy array:
        buffer = ISMBuffer.open('foo')
        arr = buffer.asarray()
        arr.fill(353)
        send_arr_to_other_process(arr)
        """
        return cls(name, manager=manager)

    @classmethod
    def new(cls, name, shape, dtype, order='C', permissions=0o600, manager=False):
        """Create a shared interprocess numpy array:
        buffer = ISMBuffer.new('foo', (10,10), int)
        arr = buffer.asarray()
        arr.fill(353)
        send_arr_to_other_process(arr)"""
        dtype = numpy.dtype(dtype)
        dtype_str = numpy.lib.format.dtype_to_descr(dtype)
        size = numpy.multiply.reduce(shape, dtype=int) * dtype.itemsize
        descr = json.dumps((dtype_str, shape, order)).encode('ascii')
        return cls(name, create=True, permissions=permissions, size=size, descr=descr, manager=manager)

    def __init__(self, name, create, permissions, size, descr, manager):
        pass

    def asarray(self):
        array = numpy.array(self, dtype=numpy.uint8, copy=False)
        if self.descr:
            dtype, shape, order = json.loads(self.descr.decode('ascii'))
            array = numpy.ndarray(shape, dtype=dtype, order=order, buffer=array)
        return array
