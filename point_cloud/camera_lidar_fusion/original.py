import time

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import cv2

import open3d as o3d
import open3d.core as o3c
import matplotlib.pyplot as plt
cmap = plt.cm.jet



def get_colored_depth(depth):
    """
    渲染深度图, depth_colorize函数的封装
    :param depth:  numpy.ndarray `H x W`
    :return:       numpy.ndarray `H x W x C'  RGB
    """
    if len(depth.shape) == 3:
        depth = depth.squeeze()
    colored_depth = depth_colorize(depth).astype(np.uint8)
    colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR)
    return colored_depth


def depth_colorize(depth):
    """
    深度图着色渲染
    :param depth: numpy.ndarray `H x W`
    :return: numpy.ndarray `H x W x C'  RGB
    example:
    n = np.arange(90000).reshape((300, 300))
    colored = depth_colorize(n).astype(np.uint8)
    colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    cv2.imshow('test', colored)
    cv2.waitKey()
    """
    assert depth.ndim == 2, 'depth image shape need to be `H x W`.'
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth



def read_calib(calib_path):
    """
    读取kitti数据集标定文件
    下载的彩色图像是左边相机的图像, 所以要用P2
    extrinsic = np.matmul(R0, lidar2camera)
    intrinsic = P2
    P中包含第i个相机到0号摄像头的距离偏移(x方向)
    extrinsic变换后的点云是投影到编号为0的相机(参考相机)坐标系中并修正后的点
    intrinsic(P2)变换后可以投影到左边相机图像上
    P0, P1, P2, P3分别代表左边灰度相机，右边灰度相机，左边彩色相机，右边彩色相机
    :return: P0-P3 numpy.ndarray           `3 x 4`
             R0 numpy.ndarray              `4 x 4`
             lidar2camera numpy.ndarray    `4 x 4`
             imu2lidar numpy.ndarray       `4 x 4`

    >>> P0, P1, P2, P3, R0, lidar2camera_m, imu2lidar_m = read_calib(calib_path)
    >>> extrinsic_m = np.matmul(R0, lidar2camera_m)
    >>> intrinsic_m = P2
    """
    with open(calib_path, 'r') as f:
        raw = f.readlines()
    P0 = np.array(list(map(float, raw[0].split()[1:]))).reshape((3, 4))
    P1 = np.array(list(map(float, raw[1].split()[1:]))).reshape((3, 4))
    P2 = np.array(list(map(float, raw[2].split()[1:]))).reshape((3, 4))
    P3 = np.array(list(map(float, raw[3].split()[1:]))).reshape((3, 4))
    R0 = np.array(list(map(float, raw[4].split()[1:]))).reshape((3, 3))
    R0 = np.hstack((R0, np.array([[0], [0], [0]])))
    R0 = np.vstack((R0, np.array([0, 0, 0, 1])))
    lidar2camera_m = np.array(list(map(float, raw[5].split()[1:]))).reshape((3, 4))
    lidar2camera_m = np.vstack((lidar2camera_m, np.array([0, 0, 0, 1])))
    imu2lidar_m = np.array(list(map(float, raw[6].split()[1:]))).reshape((3, 4))
    imu2lidar_m = np.vstack((imu2lidar_m, np.array([0, 0, 0, 1])))
    return P0, P1, P2, P3, R0, lidar2camera_m, imu2lidar_m



def read_bin(bin_path, intensity=False):
    """
    读取kitti bin格式文件点云
    :param bin_path:   点云路径
    :param intensity:  是否要强度
    :return:           numpy.ndarray `N x 3` or `N x 4`
    """
    lidar_points = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    if not intensity:
        lidar_points = lidar_points[:, :3]
    return lidar_points


def o3d_points_to_depth(points, height, width, intrinsic, extrinsic):
    """
    三维点云投影生成深度图
    @param points:       np.ndarray  [N, 3]
    @param height:       int
    @param width:        int
    @param intrinsic:    np.ndarray  [3, 3]
    @param extrinsic:    np.ndarray  [4, 4]
    @return:             np.ndarray  [H, W]    float32
    """
    pcd = o3d.t.geometry.PointCloud(o3c.Tensor(points, dtype=o3c.float32))
    intrinsic = o3c.Tensor(intrinsic, dtype=o3c.float32)
    extrinsic = o3c.Tensor(extrinsic, dtype=o3c.float32)

    o3d_depth_image = pcd.project_to_depth_image(width=width, height=height,
                                                 intrinsics=intrinsic, extrinsics=extrinsic,
                                                 depth_scale=1, depth_max=100)
    return np.asarray(o3d_depth_image).squeeze()


def o3d_depth_to_points(depth, intrinsic, extrinsic):
    """
    深度图反投影生成点云
    @param depth:         np.ndarray  [H, W]   float32
    @param intrinsic:     np.ndarray  [3, 3]
    @param extrinsic:     np.ndarray  [4, 4]
    @return:              np.ndarray  [N, 3]
    """
    intrinsic = o3c.Tensor(intrinsic, dtype=o3c.float32)
    extrinsic = o3c.Tensor(extrinsic, dtype=o3c.float32)

    o3d_depth_image = o3d.t.geometry.Image(o3c.Tensor(depth, dtype=o3c.float32))
    pcd = o3d.t.geometry.PointCloud.create_from_depth_image(depth=o3d_depth_image,
                                                            intrinsics=intrinsic, extrinsics=extrinsic,
                                                            depth_scale=1, depth_max=100)
    positions = pcd.point['positions']
    return positions.numpy()


def o3d_rgbd_to_points(rgb, depth, intrinsic, extrinsic):
    """
    rgbd图像反投影生成点云
    @param rgb:           np.ndarray  [H, W, 3]  uint8   RGB channel index
    @param depth:         np.ndarray  [H, W]     float32
    @param intrinsic:     np.ndarray  [3, 3]
    @param extrinsic:     np.ndarray  [4, 4]
    @return:
        positions:        np.ndarray  [N, 3]
        colors:           np.ndarray  [N, 3]   [0.0, 1.0]
    """
    intrinsic = o3c.Tensor(intrinsic, dtype=o3c.float32)
    extrinsic = o3c.Tensor(extrinsic, dtype=o3c.float32)

    rgbd_image = o3d.t.geometry.RGBDImage(
        color=o3d.t.geometry.Image(o3c.Tensor(rgb, dtype=o3c.uint8)),
        depth=o3d.t.geometry.Image(o3c.Tensor(depth, dtype=o3c.float32)),
        aligned=True)
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                           intrinsics=intrinsic, extrinsics=extrinsic,
                                                           depth_scale=1, depth_max=100)
    positions = pcd.point['positions']
    colors = pcd.point['colors']
    return positions.numpy(), colors.numpy()


def o3d_points_to_rgbd(points, colors, height, width, intrinsic, extrinsic):
    """
    着色点云投影生成rgbd图像
    @param points:       np.ndarray  [N, 3]
    @param colors:       np.ndarray  [N, 3]
    @param height:       int
    @param width:        int
    @param intrinsic:    np.ndarray  [3, 3]
    @param extrinsic:    np.ndarray  [4, 4]
    @return:
        color_image      np.ndarray  [H, W, 3]  uint8   RGB channel index
        depth_image      np.ndarray  [H, W]     float32
    """
    pcd = o3d.t.geometry.PointCloud()
    pcd.point['positions'] = o3c.Tensor(points, dtype=o3c.float32)
    pcd.point['colors'] = o3c.Tensor(colors, dtype=o3c.float32)
    intrinsic = o3c.Tensor(intrinsic, dtype=o3c.float32)
    extrinsic = o3c.Tensor(extrinsic, dtype=o3c.float32)

    o3d_rgbd_image = pcd.project_to_rgbd_image(width=width, height=height,
                                               intrinsics=intrinsic, extrinsics=extrinsic,
                                               depth_scale=1, depth_max=100)
    color_image = np.asarray(np.asarray(o3d_rgbd_image.color) * 255, dtype=np.uint8)
    depth_image = np.asarray(o3d_rgbd_image.depth).squeeze()
    return color_image, depth_image


if __name__ == '__main__': 
    image_path = '../data/image_2/000043.png'
    bin_path = '../data/velodyne/000043.bin'
    calib_path = '../data/calib/000043.txt'
    point_in_lidar = read_bin(bin_path)
    color_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    _, _, P2, _, R0, lidar2camera_matrix, _ = read_calib(calib_path)
    intrinsic = P2[:, :3]  # 内参
    extrinsic = np.matmul(R0, lidar2camera_matrix)  # 雷达到相机外参
    height, width = color_image.shape[:2]  # 图像高和宽

    depth1 = o3d_points_to_depth(point_in_lidar, height, width, intrinsic, extrinsic)
    # new_points = o3d_depth_to_points(depth, intrinsic, extrinsic)

    # colored_depth = get_colored_depth(depth1)
    # cv2.imshow('colored_depth', colored_depth)
    # cv2.waitKey()

    new_points, colors = o3d_rgbd_to_points(color_image, depth1, intrinsic, extrinsic)
    color_image, depth2 = o3d_points_to_rgbd(new_points, colors, height, width, intrinsic, extrinsic)
    colored_depth = get_colored_depth(depth2)
    
    # Displaying the lidar point cloud
    app = pg.mkQApp('main')
    widget = gl.GLViewWidget()
    point_size = np.zeros(new_points.shape[0], dtype=np.float16) + 0.1
    # new_points[:, 2] += 10  # Raise the rendered point cloud on the Z-axis
    
    points_item1 = gl.GLScatterPlotItem(pos=new_points, size=point_size, color=colors, pxMode=False)
    
    # Adding the unrendered lidar point cloud
    # Assuming point_in_lidar is the raw point cloud positional data
    # Using a default color (choice of darker gray or other colors)
    point_size_unrendered = np.zeros(point_in_lidar.shape[0], dtype=np.float16) + 0.1
    unrendered_color = np.full((point_in_lidar.shape[0], 4), [0.5, 0.5, 0.5, 1.0])  # Dark gray, opacity 1.0

    points_item2 = gl.GLScatterPlotItem(pos=point_in_lidar, size=point_size_unrendered, color=unrendered_color, pxMode=False)

    # Adding the unrendered point cloud to the view, placing it above the rendered point cloud
    widget.addItem(points_item2)
    # widget.addItem(points_item1)
    widget.show()
    pg.exec()