import tensorrt as trt
import ctypes
import torch
from tqdm import tqdm

#  engine_path = '/home/mr/Workspace/mdet/log/centerpoint_pp_waymo_3cls_small_range_gtaug/version_0/model_fp16.trt'
engine_path = '/tmp/model_fp32.trt'
plugin_path = '/home/mr/Workspace/avengers/perception_node2/install/tensorrt_plugin/lib/libtensorrt_plugin.so'

ctypes.CDLL(plugin_path)

logger = trt.Logger(trt.Logger.ERROR)

with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    dtype = engine.get_binding_dtype(i)
    shape = engine.get_binding_shape(i)
    if engine.binding_is_input(i):
        print(f'input: {name}, {shape}, {dtype}')
    else:
        print(f'output: {name}, {shape}, {dtype}')

points = torch.rand((10000, 4), dtype=torch.float32).cuda()
heatmap = torch.empty((1, 3, 372, 372), dtype=torch.float32).cuda()
valid = torch.empty((), dtype=torch.float32).cuda()
boxes = torch.empty((128, 8), dtype=torch.float32).cuda()
label = torch.empty((128,), dtype=torch.int32).cuda()
score = torch.empty((128,), dtype=torch.float32).cuda()

bindings = [points.data_ptr(), heatmap.data_ptr(), valid.data_ptr(),
            boxes.data_ptr(), label.data_ptr(), score.data_ptr()]

context.set_binding_shape(0, points.shape)
print(context.get_binding_shape(0))

print(label)

ret = context.execute_v2(bindings)

print('------------------------------')
print(label)
