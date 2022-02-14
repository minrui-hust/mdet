# Model zoo

## One-stage

## pillar-based
| commit | config | log | Veh | Cyc | Ped | MAPH |
|--------|--------|-----|------------|------------|------------|-------------|
| dc6b5d2 | waymo/center_point/centerpoint_pp_waymoD5_3cls_lowreso.py | [log](oss://ld-sharing/model/waymo/centerpoint_pp_waymoD5_3cls_lowreso/version_0) | 55.22 | 20.47 | 42.22 | 39.30 |
| dc6b5d2 | waymo/center_point/centerpoint_pp_waymo_3cls_baseline.py | [log](oss://ld-sharing/model/waymo/centerpoint_pp_waymo_3cls_baseline/version_0) | 63.95 | 54.12 | 58.56 | 58.88 |
| dc6b5d2 | waymo/center_point/centerpoint_pp_waymoD5_3cls_baseline.py | [log](oss://ld-sharing/model/waymo/centerpoint_pp_waymoD5_3cls_baseline/version_0) | 60.74 | 44.75 | 53.13 | 52.87 |
|         | waymo/center_point/centerpoint_pp_waymoD5_3cls_gaussian2d.py | [log](log/centerpoint_pp_waymoD5_3cls_gaussian2d/version_0) | 60.68 | 46.83 | 55.62 | 54.38  |

with amp
| commit | config | log | Veh | Cyc | Ped | MAPH |
|--------|--------|-----|------------|------------|------------|-------------|
| | waymo/center_point/centerpoint_pp_waymoD5_3cls_gtaug.py | [log](log/centerpoint_pp_waymoD5_3cls_gtaug/version_0) | 60.56  | 59.82  | 57.05  | 59.14  |
| | waymo/center_point/centerpoint_pp_waymoD5_3cls_gt_local_aug.py | [log](log/centerpoint_pp_waymoD5_3cls_gt_local_aug/version_0) | 60.92  | 59.72  | 53.21 | 57.95 |



## voxel-based

