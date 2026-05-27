#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _basic_class.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/7/25 11:31
import ast
import collections
import csv
import itertools
import json
import math
from typing import List, Tuple, Dict, Set, Optional, FrozenSet, Iterable
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


"""
define the basic data structure and operations used in various companion pattern (e.g., moving cluster, convoy, swarm) detection algorithms. 
"""


# 用于定义伴随模式挖掘中的基础数据结构与常用操作
# ================================
# 1. Trajectory Point
# ================================
class TrajectoryPoint:
    """Represents a single point at certain time in a moving object's trajectory"""

    def __init__(self, x, y, time, oid='0', truth=-1, probability: float = 1.0, direction=1):
        self.x = x  # 空间位置X
        self.y = y  # 空间位置Y
        self.time = time  # 时间戳
        self.oid = oid  # 对象ID
        self.visited = False
        self.truth = truth  # 该移动对象轨迹点所在时刻的真实标签 -1 表示没有所属的伴随群体, 用于模式挖掘结果评价时使用
        self.probability = probability  # confidence of this point (1.0 for observed points).
        self.direction = direction

    def distance_to(self, other) -> float:
        """Calculate Euclidean distance to another point"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def haversine_distance_to(self, other) -> float:
        """Calculate Haversine distance to another point"""
        lon1, lat1, lon2, lat2 = map(math.radians, [float(self.x), float(self.y), float(other.x), float(other.y)])

        # haversine 公式
        d_lng = lon2 - lon1
        d_lat = lat2 - lat1
        a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # 地球平均半径，单位为公里
        dist = c * r * 1000  # *1000, 单位为米
        return dist

    def __repr__(self):
        return f"({self.oid}, t={self.time}, x={self.x}, y={self.y}, dir={self.direction})"


# ================================
# 2. Trajectory (List of Points)
# ================================
# @dataclass
class Trajectory:
    # object_id: int
    # points: Dict[int, TrajectoryPoint] = field(init=False)   # key = time
    # visited: bool = field(init=False)

    # def __post_init__(self):
    #     self.visited = False
    #     self.points = {}
    """Represents the complete trajectory of a moving object"""

    def __init__(self, oid, points=None):
        self.oid = oid  # trajectory id = oid
        self.visited = False
        if points is not None:
            self.points_list: List[Tuple[int, TrajectoryPoint]] = points
            self.points: Dict[int, TrajectoryPoint] = points  # key = time
        else:
            self.points: Dict[int, TrajectoryPoint] = {}
            self.points_list: List[Tuple[int, TrajectoryPoint]] = []

    def add_point(self, point: TrajectoryPoint):
        self.points[point.time] = point
        self.points_list.append((point.time, point))

    def get_point(self, time: int):
        return self.points.get(time, None)

    def get_time_range(self) -> Tuple[int, int]:
        return min(self.points.keys()), max(self.points.keys())

    def get_segment_at_time(self, t: int) -> Optional[Tuple[TrajectoryPoint, TrajectoryPoint]]:
        """Get the line segment that covers time t"""
        for i in range(len(self.points_list) - 1):
            if self.points_list[i][1].time <= t <= self.points_list[i + 1][1].time:
                return (self.points_list[i][1], self.points_list[i + 1][1])
        return None

    def points_in_interval(self, t0: int, t1: float) -> List[TrajectoryPoint]:
        """返回闭区间[t0, t1]内的点（可能空）"""
        return [p for p in self.points.values() if t0 <= p.time <= t1]

    def interpolate_point(self, t: int) -> TrajectoryPoint:
        """Linearly interpolate a point at time t"""
        segment = self.get_segment_at_time(t)
        if segment is None:
            raise ValueError(f"Cannot interpolate at time {t} - outside trajectory range")

        start, end = segment
        ratio = (t - start.time) / (end.time - start.time)
        x = start.x + ratio * (end.x - start.x)
        y = start.y + ratio * (end.y - start.y)
        return TrajectoryPoint(x, y, t, start.oid)

    @property
    def start_time(self):
        # return max(self.points.keys())
        return self.points_list[0][1].time

    @property
    def end_time(self):
        # return min(self.points.keys())
        return self.points_list[-1][1].time

    def __repr__(self):
        return f"Trajectory({self.oid}, {len(self.points)} points)"


# ================================
# 3. Moving Object (with Trajectory)
# ================================
@dataclass
class MovingObject:
    # oid: str
    # trajectory: Trajectory = field(init=False)
    """Represents a moving object with its trajectory"""

    def __init__(self, oid: str):
        self.oid = oid
        self.trajectory = Trajectory(oid)

    # def __post_init__(self):

    def add_point(self, point: TrajectoryPoint):
        self.trajectory.add_point(point)

    def get_position(self, time: int) -> TrajectoryPoint:
        return self.trajectory.get_point(time)

    def get_interval_segment(self, t0: int, t1: int) -> Optional[Tuple[float, float, float, float]]:
        """
        对在 [t0,t1] 区间内的轨迹点拟合一条直线（起点->终点向量），
        返回拟合半直线的起点 (x0,y0) 和方向向量 (dx, dy)（未单位化）。
        如果该对象在该区间没有足够点（至少2个），返回 None。
        """
        pts = self.trajectory.points_in_interval(t0, t1)
        if len(pts) < 2:
            return None
        # 简单方法：使用区间内第一个点和最后一个点构成向量（工程实现，便于解释与复现）
        p0 = pts[0]
        p1 = pts[1]
        dx = p1.x - p0.x
        dy = p1.y - p0.y
        # 若零向量（无明显移动）, 则可视为没有有效方向
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None
        return (p0.x, p0.y, dx, dy)

    def __repr__(self):
        return f"Object({self.oid})"


# ================================
# 4. Cluster (at a single timestamp)
# ================================
class Cluster:
    def __init__(self, cid: str, t: int, members: List[str], occ: int = 1, objects: Dict[str, MovingObject] = None):
        self.cid = cid
        self.time = t
        self.members = members
        self.objects = objects
        self.assigned = False
        self.closed = False
        self.mbr: Optional[Tuple[float, float, float, float]] = None  # (min_x, min_y, max_x, max_y)
        self.grid_cells: Optional[Set[Tuple[int, int]]] = None  # For grid-based indexing
        self.occ = occ  # number of occurrences (default 1) -usually 1 unless duplicates exist.
        self.center: Tuple[float, float] = (0.0, 0.0)  # center of the cluster
        self.center = TrajectoryPoint(0.0, 0.0, self.time, 'center')

    def size(self):
        return len(self.members)

    def add_member(self, member):
        """Add a moving object to this cluster"""
        self.members.append(member)

    def remove_member(self, member):
        self.members.remove(member)

    def hausdorff_distance_to(self, other: 'Cluster', objects: Dict[str, MovingObject], metric='hausdorff', ):
        """
        Calculate the Hausdorff distance between two clusters
        Hausdorff distance is the maximum of:
            1. max(min distance from any point in cluster1 to cluster2)
            2. max(min distance from any point in cluster2 to cluster1)
        """
        cluster_a_members = self.members
        cluster_b_members = other.members

        def directed_hausdorff(cluster_a, a_t, cluster_b, b_t):
            max_min_dist = -math.inf
            for obj_id in cluster_a:
                min_dist = math.inf
                pt1 = objects[obj_id].get_position(a_t)
                for other_obj_id in cluster_b:
                    pt2 = objects[other_obj_id].get_position(b_t)
                    dist = pt1.distance_to(pt2)
                    if dist < min_dist:
                        min_dist = dist
                        if min_dist == 0:
                            break
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
            return max_min_dist

        return max(directed_hausdorff(cluster_a_members, self.time, cluster_b_members, other.time),
                   directed_hausdorff(cluster_b_members, other.time, cluster_a_members, self.time))

    def is_contained_by(self, super_cluster: 'Cluster') -> bool:
        """
        判断包含关系 q ⊆c s (Cluster Containment Match) [cite: 93, 94]。
        即当前聚类中的所有对象是否包含在超聚类中。
        :param super_cluster:
        :return:
        """
        return set(self.members).issubset(set(super_cluster.members))

    def contains(self, sub_cluster: 'Cluster') -> bool:
        """
        判断包含关系 q ⊆c s (Cluster Containment Match) [cite: 93, 94]。
        即当前聚类中的所有对象是否包含在超聚类中。
        :param super_cluster:
        :return:
        """
        return set(self.members).issuperset(set(sub_cluster.members))


    def calculate_center(self) -> TrajectoryPoint:
        """计算聚类中心点"""
        if not self.members:
            return TrajectoryPoint(0.0, 0.0, self.time, '-1')

        sum_x = sum(obj.get_position(self.time).x for obj in self.objects.values())
        sum_y = sum(obj.get_position(self.time).y for obj in self.objects)
        return TrajectoryPoint(sum_x / len(self.objects), sum_y / len(self.objects), self.time, 'center')

    def calculate_radius(self) -> float:
        center = self.center
        max_distance = 0.0
        for obj in self.objects.values():
            distance = center.distance_to(obj.get_position(self.time))
            max_distance = max(max_distance, distance)
        return max_distance

    def intersection_size(self, other: 'Cluster') -> int:
        """计算与另一个簇的交集大小"""
        return len(set(self.members) & set(other.members))

    def __contains__(self, object_id: str) -> bool:
        return object_id in self.members

    def __len__(self):
        return len(self.members)

    def __hash__(self):
        return hash(self.cid)

    def __eq__(self, other):
        return isinstance(other, Cluster) and self.cid == other.cid

    def __repr__(self):
        return f"Cluster(cid={self.cid}, t={self.time}, members={self.members})"


# ================================
# 5. Snapshot (all data at a single timestamp)
# ================================
class Snapshot:
    """Represents a snapshot of all moving object points and clusters at a specific timestamp"""

    def __init__(self, time: int):
        self.time = time
        self.points: Dict[str, TrajectoryPoint] = {}  # 对象ID -> 轨迹点
        self.clusters: List[Cluster] = []
        self.grid_index = None
        self.clusters_map: Dict[str, str] = {}

    def add_point(self, point: TrajectoryPoint):
        self.points[point.oid] = point

    def add_cluster(self, cluster: Cluster):
        self.clusters.append(cluster)

    def get_neighbors(self, point: TrajectoryPoint, radius: float) -> Set[str]:
        """ Get all object_ids within radius of the given point."""
        neighbors = set()
        for obj_id, p in self.points.items():
            if p.distance_to(point) <= radius:
                neighbors.add(obj_id)
        return neighbors

    def cluster_snapshot(self, snapshot: 'Snapshot', eps: float, minPts: int, metric='precomputed') -> Dict[
        str, Cluster]:

        objs = list(snapshot.points.keys())
        points = list(snapshot.points.values())
        if len(points) < minPts:
            return {}
        data = np.array([[p.x, p.y] for p in points])
        if metric == 'precomputed':
            matrix = np.zeros((len(objs), len(objs)))
            for i in range(len(objs)):
                o1_pt = snapshot.points[objs[i]]
                for j in range(i + 1, len(data)):
                    o2_pt = snapshot.points[objs[j]]
                    spatial_dist = np.sqrt((o1_pt.x - o2_pt.x) ** 2 + (o1_pt.y - o2_pt.y) ** 2)

                    matrix[i][j] = spatial_dist
            matrix = matrix + matrix.T

            labels = DBSCAN(eps=eps, min_samples=minPts, metric=metric).fit(matrix).labels_
        else:
            labels = DBSCAN(eps=eps, min_samples=minPts, metric='euclidean').fit(data).labels_

        clusters: Dict[str, Cluster] = {}
        clusters_map: Dict[str, str] = {}
        for i, label in enumerate(labels):
            if label != -1:
                c_id = f'{snapshot.time}_{label}'
                if c_id not in clusters.keys():
                    cluster = Cluster(cid=c_id, t=snapshot.time, members=[points[i].oid])
                    clusters[c_id] = cluster
                else:
                    clusters[c_id].add_member(points[i].oid)
                clusters_map[points[i].oid] = c_id
        self.clusters = list(clusters.values())
        self.clusters_map = clusters_map
        return clusters

    def get_clusters_for_object(self, object_id: str) -> List[Cluster]:
        """Get all clusters containing the specified object at this timestamp"""
        return [cluster for cluster in self.clusters if object_id in cluster.members]

    def build_grid_index(self, cell_size: float):
        """Build grid index for clusters in this snapshot"""
        grid_index = defaultdict(set)
        if self.clusters:
            for cluster in self.clusters:
                if cluster.members is None:
                    continue
                points = [self.points[oid] for oid in cluster.members]
                min_x, min_y = min([pt.x for pt in points]), min([pt.y for pt in points])
                max_x, max_y = max([pt.x for pt in points]), max([pt.y for pt in points])
                # Determine which grid cells this cluster's MBR covers
                start_i, start_j = int(min_x // cell_size), int(min_y // cell_size)
                end_i, end_j = int(max_x // cell_size), int(max_y // cell_size)
                cluster_cells = set()
                for i in range(start_i, end_i + 1):
                    for j in range(start_j, end_j + 1):
                        grid_index[(i, j)].add(cluster)
                        cluster_cells.add((i, j))

                cluster.grid_cells = cluster_cells
            self.grid_index = grid_index

    def __repr__(self):
        return f"Snapshot(t={self.time}, objects={len(self.points)}, clusters={len(self.clusters)}"


# ================================
# 6. Converging node & tree (cluster containment tree)
# ================================
@dataclass
class ConvergingTreeNode:
    cluster: Cluster
    children: List['ConvergingTreeNode'] = field(default_factory=list)
    parent: Optional['ConvergingTreeNode'] = None
    level = 0  # Depth from the root node

    def add_child(self, child: 'ConvergingTreeNode'):
        child.parent = self
        self.children.append(child)
        self.level = self.level + 1

    def get_path_length(self, object_id: str) -> int:
        """获取对象在树中的路径长度"""
        path_length = 0
        current = self
        while current is not None:
            if object_id in current.cluster.members:
                path_length += 1
            current = current.parent
        return path_length

    def __repr__(self):
        return f"cluster: {self.cluster}, children: {self.children}"


class ConvergingTree:
    """汇聚树类"""

    def __init__(self, root: ConvergingTreeNode):
        # self.root: ConvergingTreeNode = ConvergingTreeNode(root_cluster)
        self.root = root
        self.leaf_nodes: Dict[int, List[ConvergingTreeNode]] = defaultdict(list)
        self.opening_time = root.cluster.time  # p.open [cite: 586]
        self.closing_time = root.cluster.time
        self.height = 0
        # 参与者路径长度统计 {oid: path_length}
        self.participator_paths: Dict[int, int] = collections.defaultdict(int)
        # 初始化路径长度：对象在树中出现的时间戳数量 [cite: 102, 575]
        for obj_id in root.cluster.members:
            self.participator_paths[int(obj_id)] = 1

    def get_height(self) -> int:
        """获取树的高度"""

        def _get_height(node: ConvergingTreeNode) -> int:
            if not node.children:
                return 0
            return 1 + max(_get_height(child) for child in node.children)

        return _get_height(self.root)

    def update_tree(self, new_match: Tuple[Cluster, Cluster]):
        """根据新的匹配关系 (sub_cluster, super_cluster) 更新树 """
        sub_cluster, super_cluster = new_match
        # 简化：只处理新匹配的 super_cluster 是当前树根节点的情况
        if super_cluster.cid != self.root.cluster.cid:
            return False
        # 1. 创建并添加子节点
        new_child_node = ConvergingTreeNode(sub_cluster)
        self.root.add_child(new_child_node)
        # 2. 更新树属性
        self.opening_time = min(self.opening_time, sub_cluster.time)
        self.closing_time = max(self.closing_time, super_cluster.time)
        self.height = max(self.height, new_child_node.level)
        # 3. 更新参与者路径长度
        for obj_id in sub_cluster.members:
            # path_length = max(当前长度, 新节点的level + 1)
            self.participator_paths[int(obj_id)] = max(self.participator_paths[int(obj_id)], new_child_node.level + 1)

        return True

    def get_participators(self, k_p: int) -> Set[str]:
        """获取满足参与度阈值k_p的参与者"""
        participators = set()

        def _traverse(node: ConvergingTreeNode):
            for child in node.children:
                _traverse(child)
            # 检查当前节点的对象
            for oid in node.cluster.members:
                path_length = node.get_path_length(object_id=oid)
                if path_length >= k_p:
                    participators.add(oid)

        _traverse(self.root)
        return participators

    # def get_participators(self, k_p: int) -> Set[int]:
    #     """获取参与者：路径长度 |path_o| 不小于 k_p 的移动对象集合 [cite: 575]"""
    #     return {obj_id for obj_id, path_length in self.participator_paths.items() if path_length >= k_p}

    def is_valid_converging(self, k_t: int, k_m: int, k_p: int) -> bool:
        """
        判断是否为有效的汇聚模式 (Converging) [cite: 579]。
            1. Height >= k_t (lifetime threshold) [cite: 583]
            2. Participator Count >= k_m (support threshold) [cite: 584]
        """
        return (self.get_height() >= k_t and
                len(self.get_participators(k_p)) >= k_m)

    def leafset_at_ts(self, time: float) -> List[ConvergingTreeNode]:
        # returns leaves whose cluster.time == time
        leaves = []

        def dfs(node):
            if not node.children:
                if node.cluster.time == time:
                    leaves.append(node)
            else:
                for c in node.children:
                    dfs(c)

        dfs(self.root)
        return leaves

    def __repr__(self):
        return f"Converging (cid={self.root})"


class ConvergingPattern:
    """汇聚模式类"""

    def __init__(self, tree: ConvergingTree, k_t: int, k_m: int, k_p: int):
        self.tree = tree
        self.k_p = k_p
        self.k_m = k_m
        self.k_t = k_t
        self.participators = self.tree.get_participators(k_p)

    def is_valid(self) -> bool:
        """检查是否为有效的汇聚模式"""
        return (self.tree.get_height() >= self.k_t and
                len(self.participators) >= self.k_m)

    @property
    def open_time(self) -> int:
        """开始时间"""
        return self.tree.root.cluster.time

    @property
    def close_time(self) -> int:
        """结束时间"""

        # 找到最晚的时间戳
        def _get_latest_time(node: ConvergingTreeNode) -> int:
            if not node.children:
                return node.cluster.time
            # 递归循环
            return max(_get_latest_time(child) for child in node.children)

        return _get_latest_time(self.tree.root)

    def __repr__(self):
        return f"ConvergingPattern (open time={self.open_time}, close_time={self.close_time}, tree={self.tree})"


class ClusterContainmentMatch:
    """集群包含匹配类"""

    def __init__(self, sub_cluster: Cluster, super_cluster: Cluster):
        self.sub_cluster = sub_cluster
        self.super_cluster = super_cluster

    def __repr__(self):
        return f"Match({self.sub_cluster.cid} ⊆ {self.super_cluster.cid})"


@dataclass
class REMOConvergence:
    pattern_id: int = 0
    members: Set[str] = field(default_factory=set)
    center: Tuple = field(default_factory=tuple)
    obj_center: Tuple = field(default_factory=tuple)
    t_start: int = 0
    t_end: int = 0

    def update_by_another(self, other: 'REMOConvergence'):
        self.members = self.members.union(other.members)
        new_center = ((self.center[0] + other.center[0]) / 2, (self.center[1] + other.center[1]) / 2)
        self.center = new_center
        new_obj_center = ((self.obj_center[0] * len(self.members) + other.obj_center[0] * len(other.members)) / (
                    len(self.members) + len(other.members)),
                          (self.obj_center[1] * len(self.members) + other.obj_center[1] * len(other.members)) / (
                                  len(self.members) + len(other.members)))
        self.obj_center = new_obj_center

    def __repr__(self):
        return f'pid={self.pattern_id}, t_start={self.t_start}, t_end={self.t_end}, members={self.members}'


class Snowball:
    """Snowball模式类"""

    def __init__(self, clusters: List[Cluster], start_time: int, end_time: int, is_positive: bool = True):
        self.clusters = clusters  # 按时间顺序排列的聚类序列

        self.start_time = start_time
        self.end_time = end_time  # initialize the time
        self.extended = False
        self.is_positive = is_positive  # default True for snowball+ False for snowball-

    def extend_pattern(self, time: int, cluster: Cluster):
        self.clusters.append(cluster)
        self.end_time = time

    def duration(self):
        return self.end_time - self.start_time

    def size_change_rate(self) -> float:
        """计算规模变化率"""
        if len(self.clusters) < 2:
            return 0.0
        last_size = self.clusters[-2].size()
        end_size = self.clusters[-1].size()
        return (end_size - last_size) / last_size

    def __repr__(self):
        return f"Snowball {self.is_positive}({self.start_time}-{self.end_time}, " \
               f"duration={self.duration()}, clusters={[{c.cid} for c in self.clusters]}," \
               f"size_change={self.size_change_rate():.2f})"


class RealGPattern:
    """real-Gpattern类，表示时间松弛的渐进移动对象簇"""
    """
    初始化real-Gpattern

    Args:
        clusters: 聚类列表，按时间顺序排列
        pattern_type: 模式类型，"increasing"（增加）或 "decreasing"（减少）
    """

    def __init__(self, clusters: List[Cluster], pattern_type: str = "inc", pattern_id: int = 0):
        self.clusters = clusters
        self.pattern_type = pattern_type
        self.timestamps = [c.time for c in clusters]
        self.timestamps.sort()
        self.start_time = self.timestamps[0]
        self.end_time = self.timestamps[-1]
        self.pattern_id: int = 0
        # 验证模式类型
        if pattern_type not in ["inc", "dec"]:
            raise ValueError("pattern_type must be 'inc' or 'dec'")

    @property
    def size(self) -> int:
        """模式大小（聚类数量）"""
        return len(self.clusters)

    @property
    def start_timestamp(self):
        """开始时间戳"""
        return self.timestamps[0]

    @property
    def end_timestamp(self):
        """结束时间戳"""
        return self.timestamps[-1]

    @property
    def duration(self):
        """持续时间"""
        return self.end_timestamp - self.start_timestamp

    def is_valid(self, min_t: int) -> bool:
        """检查模式是否有效（满足最小时间长度要求）"""
        return self.size >= min_t

    def get_object_count_sequence(self) -> List[int]:
        """获取对象数量序列"""
        return [len(cluster) for cluster in self.clusters]

    def satisfies_graduality_condition(self) -> bool:
        """检查是否满足渐进条件"""
        if self.size < 2:
            return True

        for i in range(self.size - 1):
            if self.pattern_type == "increasing":
                # 递增模式：c_i ⊆ c_{i+1} 且 |c_{i+1}| > |c_i|
                if not set(self.clusters[i].members).issubset(set(self.clusters[i + 1].members)):
                    return False
            else:
                # 递减模式：c_i ⊇ c_{i+1} 且 |c_{i+1}| < |c_i|
                if not set(self.clusters[i].members).issuperset(set(self.clusters[i + 1].members)):
                    return False

        # 检查首尾大小关系
        if self.pattern_type == "increasing":
            return len(self.clusters[-1]) > len(self.clusters[0])
        else:
            return len(self.clusters[-1]) < len(self.clusters[0])

    def __str__(self):
        obj_counts = self.get_object_count_sequence()
        return f"RealGPattern({self.pattern_type}, size={self.size}, objects={obj_counts})"

    def __len__(self):
        return self.size

    def __repr__(self):
        return f'RealGPattern(pid={self.pattern_id}, p_type={self.pattern_type}, s_time={self.start_time}, ' \
               f'e_time={self.end_time}, clusters={self.clusters})'


class ConvergingMPattern:
    """汇聚模式类"""

    def __init__(self, center: Tuple[float, float], start_time: int, end_time: int):
        self.center = center   # 汇聚中心点
        self.start_time = start_time
        self.end_time = end_time
        self.groups: Dict[int, Set[str]] = {}  # timestamp -> object_ids(members)

    def add_group(self, timestamp: int, object_ids: Set[str]):
        """添加时间戳的汇聚群体"""
        self.groups[timestamp] = object_ids

    def duration(self) -> int:
        """计算模式持续时间"""
        return self.end_time - self.start_time + 1

    def __repr__(self):
        return f"ConvergingMPattern(center={self.center}, duration={self.duration()}, " \
               f"time_range=({self.start_time}, {self.end_time}), groups={self.groups})"


# ================================
# 9. Snapshot生成工具
# ================================
class SnapshotBuilder:
    @staticmethod
    def generate_snapshots(objects: Dict[str, 'MovingObject']) -> Dict[int, 'Snapshot']:
        snapshots: Dict[int, Snapshot] = defaultdict(lambda: None)
        for obj in objects.values():
            for time, point in obj.trajectory.points.items():
                if snapshots[time] is None:
                    snapshots[time] = Snapshot(time)
                snapshots[time].add_point(point)
        return dict(sorted(snapshots.items()))


# ================================
# 10. TrajectoryLoader (CSV 轨迹文件读取器)
# ================================
class TrajectoryLoader:
    @staticmethod
    def load_from_csv(filepath: str) -> Dict[str, MovingObject]:
        objects: Dict[str, MovingObject] = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                # oid, x, y, t = row[7], float(row[1]), float(row[2]), int(row[14])
                oid, x, y, t = row[7], float(row[15]), float(row[16]), int(row[14])
                point = TrajectoryPoint(x=x, y=y, time=t, oid=oid)
                if oid not in objects:
                    objects[oid] = MovingObject(oid)
                objects[oid].add_point(point)
        return objects

    @staticmethod
    def load_from_brinkhoff_csv(filepath: str) -> Dict[str, MovingObject]:
        objects: Dict[str, MovingObject] = {}
        df = pd.read_csv(filepath, sep="\t", header=None)
        for index, row in df.iterrows():
            oid, x, y, t, direction = row[1], float(row[5]), float(row[6]), int(row[4]), ast.literal_eval(row[10])
            point = TrajectoryPoint(x=x, y=y, time=t, oid=oid, direction=direction)
            if oid not in objects:
                objects[oid] = MovingObject(oid)
            objects[oid].add_point(point)
        return objects

    @staticmethod
    def load_from_csv_case2(filepath: str) -> Dict[str, MovingObject]:
        objects: Dict[str, MovingObject] = {}
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                oid, x, y, t = row[1], float(row[2]), float(row[3]), int(row[0])
                point = TrajectoryPoint(x, y, t, oid)
                if oid not in objects:
                    objects[oid] = MovingObject(oid)
                objects[oid].add_point(point)
        return objects

    @staticmethod
    def load_from_shp(filepath: str) -> Dict[str, MovingObject]:
        objects: Dict[str, MovingObject] = {}
        gdf = read_shapefile(filepath)
        for row in gdf.iterrows():
            try:
                oid, x, y, t, truth = row[1].oid, float(row[1].geometry.x), float(row[1].geometry.y), int(row[1].t), row[
                1].truth
            except:
                oid, x, y, t, truth = row[1].oid, float(row[1].geometry.x), float(row[1].geometry.y), int(row[1].time), \
                                      row[1].truth

            point = TrajectoryPoint(x, y, t, oid, truth)
            if oid not in objects:
                objects[oid] = MovingObject(oid)
            objects[oid].add_point(point)
        return objects


def read_shapefile(filename, encoding='latin1'):
    import geopandas as gpd
    data = gpd.GeoDataFrame.from_file(filename, encoding=encoding)
    return data