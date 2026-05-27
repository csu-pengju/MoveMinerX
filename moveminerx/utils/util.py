#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: util.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/4/21 21:39
import csv
import math
import re
from datetime import datetime

import networkx as nx
import scipy.io as scio
import h5py
import numpy as np
import pytz
import geopandas as gpd


def save_shapefile(filename, data, encoding='utf-8'):
    data.to_file(filename, encoding=encoding)


def read_shapefile(filename, encoding='latin1'):
    data = gpd.GeoDataFrame.from_file(filename, encoding=encoding)
    return data


def save_list(data, filename):
    np.save(filename, data)


def read_list(filename):
    return np.load(filename, allow_pickle=True)


def save_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as fp:
        fp.write(str(data))


def load_json(filename):
    globals = {
        'nan': 0
    }
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
        data = eval(data, globals)
    return data


def parse(filename, isDirected):
    reader = csv.reader(open(filename, 'r'), delimiter=',')
    data = [row for row in reader]

    print("Reading and parsing the data into memory...")
    if isDirected:
        return parse_directed(data)
    else:
        return parse_undirected(data)


def parse_undirected(data):
    G = nx.Graph()
    nodes = set([row[0] for row in data])
    edges = [(row[0], row[2]) for row in data]

    num_nodes = len(nodes)
    rank = 1 / float(num_nodes)
    G.add_nodes_from(nodes, rank=rank)
    G.add_edges_from(edges)
    pr = nx.pagerank(G, alpha=1)
    return G


def parse_directed(data):
    DG = nx.DiGraph()

    for i, row in enumerate(data):

        node_a = format_key(row[0])
        node_b = format_key(row[2])
        val_a = digits(row[1])
        val_b = digits(row[3])

        DG.add_edge(node_a, node_b)
        if val_a >= val_b:
            DG.add_path([node_a, node_b])
        else:
            DG.add_path([node_b, node_a])

    return DG


def digits(val):
    return int(re.sub(r"\D", "", val))


def format_key(key):
    key = key.strip()
    if key.startswith('"') and key.endswith('"'):
        key = key[1:-1]
    return key


def print_results(f, method, results):
    print(method)


def degree_to_meter(degree):
    return degree * (2 * math.pi * 6371004) / 360


def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_haversine_distance(lon1, lat1, lon2, lat2):
    # " 经度: lon ; 纬度: lat "
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)])

    # haversine 公式
    d_lng = lon2 - lon1
    d_lat = lat2 - lat1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    dist = c * r * 1000  # *1000, 单位为米
    return dist


def to_timestamp(t, origin_format="%Y-%m-%d %H:%M:%S", target_format=None):
    dt_object = datetime.strptime(t, origin_format)
    timestamp = dt_object.timestamp()
    return timestamp


def meter_to_degree(meter):
    # 千米转为度
    cilometter = 0.0089932202929999989 // 1

    degree = meter / (2 * math.pi * 6371004) * 360
    return degree


def get_cluster_ids_list(labels):
    pre_cluster_list = []
    labels_set = set(labels)
    k = len(labels_set)

    for i in range(k):
        x = []
        pre_cluster_list.append(x)

    outlier_ids = []
    for j in range(len(labels)):
        cluster_id = labels[j]
        if cluster_id >= 0:
            pre_cluster_list[cluster_id].append(j)
        if cluster_id == -1:
            outlier_ids.append(j)

    cluster_list = []
    for cluster in pre_cluster_list:
        if len(cluster) > 1:  # 去除小于两条的轨迹簇
            cluster_list.append(cluster)

    return cluster_list, outlier_ids


def calculate_distance_matrix(pts):
    distance_matrix = np.zeros((len(pts), len(pts)))
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            distance = calculate_distance(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
            distance_matrix[i][j] = distance
    distance_matrix = distance_matrix + distance_matrix.T
    return distance_matrix


# def get_moving_object_points(timestamp, trajectories):
#     trajectory = trajectories[:, timestamp, :]
#     snapshot_pts = {idx: MovingObjectPoint(idx, timestamp, pt[0], pt[1]) for idx, pt in enumerate(trajectory)}
#     return snapshot_pts


def calculate_azimuth(pt1, pt2):
    """
    Calculate the azimuth between two points (in degrees)
    parameters：
    - pt1: [x1, y1], the coordinates of the first point
    - pt2: [x2, y2], the coordinates of the second point
`
    return：
    the calculated azimuth (in degrees), ranging from 0 to 360

    """

    delta_x = pt2[0] - pt1[0]
    delta_y = pt2[1] - pt1[1]

    # 计算方位角
    bearing = math.atan2(delta_y, delta_x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360  # 将负值转换为正值，范围调整为0到360

    return bearing


# def initialize_clusters(cluster_list, timestamp):
#     cluster_object_list = []
#     for cluster in cluster_list:
#         temp = Cluster(timestamp, cluster)
#         cluster_object_list.append(temp)
#
#     return cluster_object_list


def calculate_moving_object_points_distance_matrix(moving_object_pts):
    distance_matrix = np.zeros((len(moving_object_pts), len(moving_object_pts)))
    moving_object_id_list = list(moving_object_pts.keys())
    for i in range(len(moving_object_id_list)):
        pt1 = moving_object_pts[moving_object_id_list[i]]
        for j in range(i + 1, len(moving_object_id_list)):
            pt2 = moving_object_pts[moving_object_id_list[i]]
            dist = calculate_distance(pt1.x, pt1.y, pt2.x, pt2.y)
            distance_matrix[i][j] = dist
    distance_matrix = distance_matrix + distance_matrix.T
    return distance_matrix


def save_to_npy(data_dict, filename):
    """
  Saves a dictionary to a .npy file.

  Args:
    data_dict: The dictionary to save.
    filename: The name of the .npy file to create (e.g., 'my_dict.npy').
  """
    try:
        np.save(filename, data_dict)
        print(f"Dictionary saved to {filename}")
    except Exception as e:
        print(f"Error saving dictionary to {filename}: {e}")


def load_from_npy(filename):
    """
  Loads a dictionary from a .npy file.

  Args:
    filename: The name of the .npy file to load (e.g., 'my_dict.npy').

  Returns:
    The dictionary loaded from the file, or None if an error occurred.
  """
    try:
        loaded_data = np.load(filename, allow_pickle=True).item()
        if isinstance(loaded_data, dict):
            return loaded_data
        else:
            print(f"Error: File {filename} does not contain a dictionary.")
            return None
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    except Exception as e:
        print(f"Error loading dictionary from {filename}: {e}")
        return None


def convert_string_to_timestamp(date_string, timezone_str='UTC'):
    """
  Converts a date string in the format 'YYYY-MM-DD HH:MM:SS' to a timestamp (seconds since epoch).

  Args:
    date_string: The date string to convert (e.g., '2014-05-01 00:00:00').
    timezone_str:  The timezone of the date string. Defaults to UTC.  It's *crucial* to
                   know the correct timezone of the input string.  Examples: 'UTC',
                   'America/Los_Angeles', 'Asia/Shanghai'.

  Returns:
    The timestamp as a float, or None if the date string is invalid.
  """
    try:
        # 1. Parse the string into a naive datetime object
        dt_naive = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')

        # 2. Localize the naive datetime object to the specified timezone
        timezone = pytz.timezone(timezone_str)
        dt_aware = timezone.localize(dt_naive)

        # 3. Convert to UTC (recommended for timestamps)
        dt_utc = dt_aware.astimezone(pytz.utc)

        # 4. Get the timestamp
        timestamp = dt_utc.timestamp()
        return timestamp
    except ValueError as e:
        print(f"Error: Invalid date string format or timezone: {e}")
        return None
    except pytz.exceptions.UnknownTimeZoneError as e:
        print(f"Error: Unknown timezone: {e}")
        return None


def load_mat(filename):

    data = scio.loadmat(filename)
    return data


def save_matrix_to_h5(matrix, filepath):
    with h5py.File(filepath, 'w') as f:
        # 从.mat加载数据, 检查是否为对称矩阵
        is_symmetric = np.allclose(matrix, matrix.T)
        if is_symmetric:
            # 只存储上三角或下三角
            upper_triangle = matrix[np.triu_indices_from(matrix)]
            matrix = upper_triangle.flatten()
        # 创建数据集并压缩
        f.create_dataset(f'dist', data=matrix,
                         compression='gzip', compression_opts=9, shuffle=True)
        f.close()


def load_matrix_from_h5(filepath):
    # 转换为HDF5（高效压缩）
    with h5py.File(filepath, "r") as hf:  # "r" 模式表示只读
        # 遍历 HDF5 文件中的所有数据集
        data = hf[f'dist'][:]
        if data.ndim == 1:
            # 尝试重建对称矩阵
            # 计算原始矩阵的大小：n(n+1)/2 = len(data) => n^2 + n - 2*len(data) = 0
            n = int((-1 + np.sqrt(1 + 8 * len(data))) / 2)
            if n * (n + 1) // 2 == len(data):
                # 重建对称矩阵
                matrix = np.zeros((n, n))
                triu_indices = np.triu_indices(n)
                matrix[triu_indices] = data
                # 填充下三角部分
                matrix = matrix + matrix.T - np.diag(np.diag(matrix))
            else:
                # 如果无法重建为对称矩阵，保持原状
                matrix = data
        else:
            # 直接返回多维数组
            matrix = data
    return matrix


def calculate_snapshot_matrix(points, has_direction=False):
    EPS = 1e-10
    coords = np.array([[p.x, p.y] for p in points])  # n x 2
    dirs = np.array([p.direction for p in points])  # n x 2
    # ---- 空间距离 (欧氏距离矩阵) ----
    diff = coords[:, None, :] - coords[None, :, :]  # n x n x 2
    spatial_dist = np.sqrt((diff ** 2).sum(-1))  # n x n

    # ---- 方向相似度 ----
    dot = dirs @ dirs.T  # n x n
    norm = np.linalg.norm(dirs, axis=1)
    denom = norm[:, None] * norm[None, :]
    cos_sim = dot / (denom + EPS)
    direction_sim = 1 - cos_sim  # 方向差异

    # ---- 综合距离 ----
    dist = 2 / (2 - direction_sim + EPS) * spatial_dist
    return dist


def calculate_hausdorff(points1, points2):
    points1 = np.array(points1)
    points2 = np.array(points2)
    # 计算所有点对之间的距离矩阵
    n1 = len(points1)
    n2 = len(points2)

    dist_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            lon1, lat1 = points1[i]
            lon2, lat2 = points2[j]
            dist_matrix[i, j] = calculate_haversine_distance(lon1, lat1, lon2, lat2)

    # 计算Hausdorff距离
    # 从points1到points2的有向Hausdorff距离
    h1 = np.max(np.min(dist_matrix, axis=1))

    # 从points2到points1的有向Hausdorff距离
    h2 = np.max(np.min(dist_matrix, axis=0))

    # Hausdorff距离是两者的最大值
    hausdorff_dist = max(h1, h2)

    return hausdorff_dist


def get_clusters_id(trajs_id, labels):
    label_unique = list(np.unique(labels))
    clusters = {label: [] for label in label_unique}
    for idx, label in enumerate(labels):
        clusters[label].append(trajs_id[idx])
    return clusters


def add_pattern_to_groups(exist_pattern_groups: dict, new_pattern):
    """
    更新pattern_groups
    :param exist_pattern_groups:
    :param new_pattern:
    :return:
    """
    if len(exist_pattern_groups) == 0:
        pattern_count = 0
    else:
        pattern_count = max(exist_pattern_groups.keys())
    exist_pattern_groups[pattern_count + 1] = {}
    exist_pattern_groups[pattern_count + 1]['pattern_type'] = new_pattern[0]
    objects = [int(obj) for obj in list(new_pattern[1])]
    exist_pattern_groups[pattern_count + 1]['objects'] = objects
    return exist_pattern_groups


def normalize_distance(value, unit):
    if unit == "m":
        return value
    elif unit == "km":
        return value * 1000.0
    elif unit == "degree":
        # 粗略换算（赤道附近）
        return degree_to_meter(value)
        # return value * 111_000.0
    else:
        return value
