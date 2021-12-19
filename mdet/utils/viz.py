import torch
import numpy as np
import open3d as o3d


class Visualizer(object):
    r"""Online visualizer implemented with Open3d.

    Args:
        points (numpy.array, shape=[N, 3+C]): Points to visualize. The Points
            cloud is in mode of Coord3DMode.DEPTH (please refer to
            core.structures.coord_3d_mode).
        bbox3d (numpy.array, shape=[M, 7]): 3d bbox (x, y, z, dx, dy, dz, yaw)
            to visualize. The 3d bbox is in mode of Box3DMode.DEPTH with
            gravity_center (please refer to core.structures.box_3d_mode).
            Default: None.
        save_path (str): path to save visualized results. Default: None.
        points_size (int): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float]): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """

    def __init__(self):
        super().__init__()
        self.app = o3d.visualization.gui.Application.instance
        self.app.initialize()

        self.viz = o3d.visualization.O3DVisualizer("Open3D", 1920, 1080)
        self.viz.ground_plane=o3d.visualization.rendering.Scene.GroundPlane.XY
        self.viz.show_ground=True
        self.viz.show_skybox(False)

        # record pcd added
        self.pcd = None

    def add_points(self, points, point_size=3, point_color=(0.5, 0.5, 0.5)):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()

        if not isinstance(point_color, np.ndarray):
            point_color = np.array(point_color)
            if point_color.ndim == 1:
                point_color = np.broadcast_to(point_color, (points.shape[0], 3))

        points = points.copy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(point_color)

        self.viz.point_size = point_size
        self.viz.add_geometry('pointcloud', pcd)
        self.pcd = pcd


    def add_box(self, boxes, box_color=(0.8, 0.1, 0.1), box_label=None, paint_point_in_box=True):
        r'''
        Add bounding box to visualizer.
        '''

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        boxes = boxes.copy()

        if not isinstance(box_color, np.ndarray):
            box_color = np.array(box_color)
            if box_color.ndim == 1:
                box_color = np.broadcast_to(box_color, (boxes.shape[0], 3))

        for i, (box, color) in enumerate(zip(boxes, box_color)):
            center = box[0:3]
            dim = box[3:6]
            rotm = o3d.geometry.get_rotation_matrix_from_xyz(np.array([0,0,box[6]]))
            o3d_box = o3d.geometry.OrientedBoundingBox(center, rotm, dim)
            line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_box)
            line_set.paint_uniform_color(color)
            self.viz.add_geometry(f'box_{i}', line_set)
            if box_label is not None:
                self.viz.add_3d_label(center, box_label[i])

            #  change the color of points which are in box
            if paint_point_in_box and self.pcd is not None:
                indices = o3d_box.get_point_indices_within_bounding_box(self.pcd.points)
                np.asarray(self.pcd.colors)[indices] = color

        if self.pcd is not None:
            self.viz.remove_geometry('pointcloud')
            self.viz.add_geometry('pointcloud', self.pcd)


    def show(self):
        self.viz.reset_camera_to_default()
        self.app.add_window(self.viz)
        self.app.run()
