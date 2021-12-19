import os
from tqdm import tqdm
import mdet.utils.io as io


def summary(root_path, split, nsweep=5):
    anno_path = os.path.join(root_path, split, 'annos')
    frame_name_list = sort_frame(list(os.listdir(anno_path)))

    infos = []
    for cur_frame_name in tqdm(frame_name_list):
        cur_anno_path = os.path.join(root_path, split, 'annos', cur_frame_name)

        cur_anno = io.load(cur_anno_path, format='pkl')
        cur_frame_id = cur_anno['frame_id']
        cur_seq_id = cur_anno['seq_name']

        sweeps = []
        for frame_id in range(cur_frame_id, cur_frame_id - nsweep, -1):
            frame_id = max(frame_id, 0)
            anno_path = os.path.join(
                root_path, split, 'annos', f'{cur_seq_id}-{frame_id}.pkl')
            pcd_path = os.path.join(
                root_path, split, 'pcds', f'{cur_seq_id}-{frame_id}.pkl')
            anno = io.load(anno_path, format='pkl')

            sweep = {
                'timestamp': anno['timestamp'],
                'tf_map_vehicle': anno['tf_map_vehicle'],
                'pcd_path': pcd_path,
            }
            sweeps.append(sweep)

        info = {
            'anno_path': os.path.join(root_path, split, 'annos', f'{cur_seq_id}-{cur_frame_id}.pkl'),
            'sweeps': sweeps,
        }
        infos.append(info)

    io.dump(infos, os.path.join(root_path, f'{split}_info.pkl'), format='pkl')


def sort_frame(frame_name_list):
    seq_frame_list = []
    for frame_name in frame_name_list:
        frame_name = os.path.splitext(frame_name)[0]
        seq, frame = frame_name.split('-')
        seq_frame_list.append((seq, int(frame)))
    seq_frame_list.sort()

    return [f'{p[0]}-{p[1]}.pkl' for p in seq_frame_list]
