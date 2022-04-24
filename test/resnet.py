import torch
import torch.nn as nn
import torchvision.models as mm
from torchvision.models._utils import IntermediateLayerGetter
from time import time

interm_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
res50 = IntermediateLayerGetter(
    mm.resnet34(), return_layers=interm_layers).half().cuda()
res50.eval()

din = torch.rand((2, 3, 720, 1280), dtype=torch.half).cuda()

tick = time()
for _ in range(200):
    dout = res50(din)
    print(dout['0'].shape, dout['1'].shape, dout['2'].shape)
print(f'dt: {time()-tick}')
