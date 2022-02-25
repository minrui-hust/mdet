import torch
from mdet.ops.iou3d.iou3d_utils import iou_bev
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

boxes_a = torch.load('tmp/pred_box.pt')
boxes_b = torch.load('tmp/gt_box.pt')
gt_iou = torch.load('tmp/iou.pt')

print(boxes_a[0])
print(boxes_b[0])
cc_iou = iou_bev(boxes_a[[0]], boxes_b[[0]])
print(cc_iou)

center_a = boxes_a[0, [0, 1]]
l_a, w_a = boxes_a[0, 2]*2, boxes_a[0, 3]*2
cos_a, sin_a = boxes_a[0, 4], boxes_a[0, 5]
dx_a = (l_a*cos_a - w_a*sin_a)/2
dy_a = (l_a*sin_a + w_a*cos_a)/2
pos_a = center_a - torch.stack([dx_a, dy_a])

center_b = boxes_b[0, [0, 1]]
l_b, w_b = boxes_b[0, 2]*2, boxes_b[0, 3]*2
cos_b, sin_b = boxes_b[0, 4], boxes_b[0, 5]
dx_b = (l_b*cos_b - w_b*sin_b)/2
dy_b = (l_b*sin_b + w_b*cos_b)/2
pos_b = center_b - torch.stack([dx_b, dy_b])


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, aspect='equal')
rect_a = Rectangle((pos_a[0].item(), pos_a[1].item()), l_a.item(), w_a.item(), angle=torch.atan2(
    sin_a, cos_a).item()*180/3.1415926, linewidth=2, edgecolor='red', facecolor='none')
rect_b = Rectangle((pos_b[0].item(), pos_b[1].item()), l_b.item(), w_b.item(), angle=torch.atan2(
    sin_b, cos_b).item()*180/3.1415926, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect_a)
ax.add_patch(rect_b)
ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])
plt.show()
