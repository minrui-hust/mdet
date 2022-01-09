"""Tool to convert Waymo Open Dataset to seperate files.
"""

import os
import os.path as osp
import multiprocessing as mp
from functools import partial

import tensorflow.compat.v2 as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils

from tqdm import tqdm

import zlib
import numpy as np

import mdet.utils.io as io


def convert(raw_root_path, out_root_path, split):
    record_path = osp.join(raw_root_path, split)
    output_path = osp.join(out_root_path, split)

    record_path_list = [osp.join(record_path, fname) for fname in os.listdir(
        record_path) if fname.endswith('.tfrecord')]
    print(f'Number of record to process: {len(record_path_list)}')

    pcd_path = os.path.join(output_path, 'pcds')
    anno_path = os.path.join(output_path, 'annos')
    os.makedirs(pcd_path, exist_ok=True)
    os.makedirs(anno_path, exist_ok=True)

    with mp.Pool(int(mp.cpu_count()/2)) as p:
        p.map(partial(convert_one, pcd_path=pcd_path, anno_path=anno_path),
              record_path_list)


def convert_one(record_path, pcd_path, anno_path):
    tf_dataset = tf.data.TFRecordDataset(record_path, compression_type='')
    for frame_id, data in tqdm(enumerate(tf_dataset)):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        points = decode_point(frame)
        annos = decode_annos(frame, frame_id)

        seq_name = annos['seq_name']

        io.dump(
            points, f'{pcd_path}/{seq_name}-{frame_id}.pkl')
        io.dump(
            annos, f'{anno_path}/{seq_name}-{frame_id}.pkl')


def decode_point(frame):
    """Decodes native waymo Frame proto to points"""
    def sort_lambda(x):
        return x.name

    lasers_with_calibration = zip(
        sorted(frame.lasers, key=sort_lambda),
        sorted(frame.context.laser_calibrations, key=sort_lambda))

    points_list = []
    for laser, calibration in lasers_with_calibration:
        points_list.extend(
            extract_points_from_range_image(laser, calibration, frame.pose))

    return np.concatenate(points_list, axis=0)


def decode_annos(frame, frame_id):

    tf_map_vehicle = np.reshape(np.array(frame.pose.transform), [4, 4])
    objects = extract_objects(frame.laser_labels)

    annos = {
        'frame_id': frame_id,
        'timestamp': frame.timestamp_micros,
        'tf_map_vehicle': tf_map_vehicle,
        'objects': objects,
        'seq_name': frame.context.name,
    }

    return annos


def extract_points_from_range_image(laser, calibration, frame_pose):
    """Decode points from lidar."""
    if laser.name != calibration.name:
        raise ValueError('Laser and calibration do not match')

    if laser.name == dataset_pb2.LaserName.TOP:
        frame_pose = tf.convert_to_tensor(
            np.reshape(np.array(frame_pose.transform), [4, 4]))

        range_image_top_pose = dataset_pb2.MatrixFloat.FromString(
            zlib.decompress(laser.ri_return1.range_image_pose_compressed))

        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0],
            range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        frame_pose = tf.expand_dims(frame_pose, axis=0)
        pixel_pose = tf.expand_dims(range_image_top_pose_tensor, axis=0)
    else:
        pixel_pose = None
        frame_pose = None
    first_return = zlib.decompress(laser.ri_return1.range_image_compressed)
    second_return = zlib.decompress(laser.ri_return2.range_image_compressed)
    points_list = []
    for range_image_str in [first_return, second_return]:
        range_image = dataset_pb2.MatrixFloat.FromString(range_image_str)
        if not calibration.beam_inclinations:
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([
                    calibration.beam_inclination_min,
                    calibration.beam_inclination_max
                ]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(calibration.beam_inclinations)
        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(calibration.extrinsic.transform),
                               [4, 4])
        range_image_tensor = tf.reshape(tf.convert_to_tensor(range_image.data),
                                        range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_cartesian = (
            range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations),
                               axis=0),
                pixel_pose=pixel_pose,
                frame_pose=frame_pose))
        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(
            tf.concat([range_image_cartesian, range_image_tensor[..., 1:4]],
                      axis=-1), tf.where(range_image_mask))
        points_list.append(points_tensor.numpy())
    return points_list


def extract_objects(laser_labels):
    """Extract objects."""
    objects = []
    for label in laser_labels:
        name = label.id
        type = label.type
        box = label.box
        num_points = label.num_lidar_points_in_box
        level = label.detection_difficulty_level
        custom_level = level

        # Difficulty level is 0 if labeler did not say this was LEVEL_2.
        # Set difficulty level of "999" for boxes with no points in box.
        if num_points <= 0:
            custom_level = 999
        if custom_level == 0:
            if num_points >= 5:
                custom_level = 1
            else:
                custom_level = 2

        objects.append({
            'name': name,
            'type': type,
            'box': np.array([box.center_x, box.center_y, box.center_z, box.length,
                             box.width, box.height, box.heading], dtype=np.float32),
            'num_points': num_points,
            'level': level,
            'custom_level': custom_level,
        })
    return objects
