import ctypes

import tensorrt as trt
import torch


class TrtModel(object):
    def __init__(self, engine_path, plugin_path=None):
        # load plugin
        if plugin_path is not None:
            ctypes.CDLL(plugin_path)

        # create execution context
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # bindings
        self.bindings = []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = self.torchDtype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            dir = 'i' if self.engine.binding_is_input(i) else 'o'
            self.bindings.append((name, shape, dtype, dir))

    def __call__(self, batch):
        output = {}
        for b in range(batch['_info_']['size']):
            bindings = []
            for i, binding in enumerate(self.bindings):
                name, shape, dtype, dir = binding
                if dir == 'i':
                    tensor = batch['input'][name][b]
                    bindings.append(tensor.data_ptr())
                    if (torch.tensor(shape) == -1).any():
                        self.context.set_binding_shape(i, tensor.shape)
                else:  # dir=='o'
                    tensor = torch.empty(tuple(shape), dtype=dtype).cuda()
                    bindings.append(tensor.data_ptr())
                    if not name in output:
                        output[name] = [tensor]
                    else:
                        output[name].append(tensor)
            self.context.execute_v2(bindings)
        return output

    def torchDtype(self, trt_dtype):
        if trt_dtype == trt.DataType.FLOAT:
            return torch.float32
        elif trt_dtype == trt.DataType.INT32:
            return torch.int32
        elif trt_dtype == trt.DataType.HALF:
            return torch.float16
        elif trt_dtype == trt.DataType.INT8:
            return torch.int8
        elif trt_dtype == trt.DataType.BOOL:
            return torch.bool
        else:
            raise NotImplementedError
