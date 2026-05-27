#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _basic_class.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 17:50

import csv
import itertools
import json
import math
from cProfile import label
from typing import List, Tuple, Dict, Set, Optional, FrozenSet, Iterable
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
from sklearn.cluster import DBSCAN

# from utils import read_shapefile
from tests.util import read_shapefile

"""
define the basic data structure and operations used in various companion pattern (e.g., moving cluster, convoy, swarm) detection algorithms. 
"""


# 用于定义伴随模式挖掘中的基础数据结构与常用操作
# ================================
# 1. Trajectory Point
# ================================
class TrajectoryPoint:
    """Represents a single point at certain time in a moving object's trajectory"""

    def __init__(self, x, y, time, oid='0', truth=-1, probability: float = 1.0):
        self.x = x  # 空间位置X
        self.y = y  # 空间位置Y
        self.time = time  # 时间戳
        self.oid = oid  # 对象ID
        self.visited = False
        self.truth = truth  # 该移动对象轨迹点所在时刻的真实标签 -1 表示没有所属的伴随群体, 用于模式挖掘结果评价时使用
        self.probability = probability  # confidence of this point (1.0 for observed points).

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
        return f"({self.oid}, t={self.time}, x={self.x}, y={self.y})"


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
        return TrajectoryPoint(start.oid, x, y, t)

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


class ClusterClass:
    def __init__(self, cid: str, t: int, members: List[str], occ: int = 1, objects: List = None):
        self.cid = cid
        self.time = t
        self.members = members
        self.objects = objects
        self.assigned = False
        self.closed = False
        self.mbr: Optional[Tuple[float, float, float, float]] = None  # (min_x, min_y, max_x, max_y)
        self.grid_cells: Optional[Set[Tuple[int, int]]] = None  # For grid-based indexing
        self.occ = occ  # number of occurrences (default 1) -usually 1 unless duplicates exist.

    def size(self):
        return len(self.members)

    def add_member(self, member):
        """Add a moving object to this cluster"""
        self.members.append(member)

    def remove_member(self, member):
        self.members.remove(member)

    def hausdorff_distance_to(self, other: 'ClusterClass', objects: Dict[str, MovingObject], metric='hausdorff', ):
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

        # Compute all pairwise distances
        # dist_matrix = np.sqrt(np.sum((points1[:, np.newaxis, :] - points2[np.newaxis, :, :]) ** 2, axis=2))
        # # Hausdorff distance is the maximum of:
        # # 1. For each point in cluster1, the minimum distance to cluster2
        # # 2. For each point in cluster2, the minimum distance to cluster1
        # h1 = np.max(np.min(dist_matrix, axis=1))
        # h2 = np.max(np.min(dist_matrix, axis=0))
        # return max(h1, h2)

        return max(directed_hausdorff(cluster_a_members, self.time, cluster_b_members, other.time),
                   directed_hausdorff(cluster_b_members, other.time, cluster_a_members, self.time))

    def __contains__(self, object_id: str) -> bool:
        return object_id in self.members

    def __repr__(self):
        return f"ClusterClass(cid={self.cid}, t={self.time}, size={self.size()})"


# ================================
# 5. Cluster (at a single timestamp)
# ================================
class Cluster:
    def __int__(self, cid: int, t: int):
        """
        :param cid:
        :param t:
        :return:
        """
        self.cid = cid
        self.time = t
        self.members = set()
        self.assigned = False
        self.closed = False
        # self.cluster_extended = 0

    def size(self):
        return len(self.members)

    def add_member(self, member):
        """Add a moving object to this cluster"""
        self.members.add(member)

    def remove_member(self, member):
        self.members.remove(member)

    def __contains__(self, object_id: str) -> bool:
        return object_id in self.members

    def __repr__(self):
        return f"Cluster(cid={self.cid}, t={self.time}, size={self.size()})"


# ================================
# 6. Snapshot (all data at a single timestamp)
# ================================
class Snapshot:
    """Represents a snapshot of all moving object points and clusters at a specific timestamp"""

    def __init__(self, time: int):
        self.time = time
        self.points: Dict[str, TrajectoryPoint] = {}  # 对象ID -> 轨迹点
        self.clusters: List[ClusterClass] = []
        self.grid_index = None
        self.clusters_map: Dict[str, str] = {}

    def add_point(self, point: TrajectoryPoint):
        self.points[point.oid] = point

    def add_cluster(self, cluster: ClusterClass):
        self.clusters.append(cluster)

    def get_neighbors(self, point: TrajectoryPoint, radius: float) -> Set[str]:
        """ Get all object_ids within radius of the given point."""
        neighbors = set()
        for obj_id, p in self.points.items():
            if p.distance_to(point) <= radius:
                neighbors.add(obj_id)
        return neighbors

    def cluster_snapshot(self, snapshot: 'Snapshot', eps: float, minPts: int, metric='precomputed') -> Dict[
        str, ClusterClass]:
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
                    # print(o1_pt.oid, o2_pt.oid, spatial_dist, o1_pt.x, o1_pt.y, o2_pt.x, o2_pt.y)
                    matrix[i][j] = spatial_dist
            matrix = matrix + matrix.T

            labels = DBSCAN(eps=eps, min_samples=minPts, metric=metric).fit(matrix).labels_
        else:
            labels = DBSCAN(eps=eps, min_samples=minPts, metric='euclidean').fit(data).labels_

        clusters: Dict[str, ClusterClass] = {}
        clusters_map: Dict[str, str] = {}
        for i, label in enumerate(labels):
            if label != -1:
                c_id = f'{snapshot.time}_{label}'
                if c_id not in clusters.keys():
                    # member_set = set()
                    # member_set.add(points[i].oid)
                    cluster = ClusterClass(cid=c_id, t=snapshot.time, members=[points[i].oid])
                    clusters[c_id] = cluster
                else:
                    # clusters[c_id].add(points[i].oid)
                    clusters[c_id].add_member(points[i].oid)
                clusters_map[points[i].oid] = c_id
        self.clusters = list(clusters.values())
        self.clusters_map = clusters_map
        return clusters

    def get_clusters_for_object(self, object_id: str) -> List[ClusterClass]:
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
# 7. TrajectorySegment (all data at a single timestamp)
# ================================
@dataclass
class LineSegment:
    """Represents a line segment in a simplified trajectory"""
    start_point: TrajectoryPoint
    end_point: TrajectoryPoint
    actual_tolerance: float  # maximum deviation from original trajectory

    @property
    def start_time(self) -> int:
        return self.start_point.time

    @property
    def end_time(self) -> int:
        return self.end_point.time

    def time_interval(self) -> Tuple[int, int]:
        return self.start_time, self.end_time

    def distance_to_point(self, point: TrajectoryPoint) -> float:
        """Calculate shortest distance from point to this line segment"""
        # Vector from start to end
        seg_vec = np.array([self.end_point.x - self.start_point.x,
                            self.end_point.y - self.start_point.y])
        # Vector from start to point
        point_vec = np.array([point.x - self.start_point.x,
                              point.y - self.start_point.y])
        seg_length = np.linalg.norm(seg_vec)
        if seg_length == 0:
            return math.sqrt((point.x - self.start_point.x) ** 2 +
                             (point.y - self.start_point.y) ** 2)

        # Projection of point_vec onto seg_vec
        projection = np.dot(point_vec, seg_vec) / seg_length
        if projection < 0:
            # Closest to start point
            return math.sqrt((point.x - self.start_point.x) ** 2 +
                             (point.y - self.start_point.y) ** 2)
        elif projection > seg_length:
            # Closest to end point
            return math.sqrt((point.x - self.end_point.x) ** 2 +
                             (point.y - self.end_point.y) ** 2)
        else:
            # Distance to line
            return abs(np.cross(seg_vec, point_vec)) / seg_length

    def distance_to_segment(self, other: 'LineSegment') -> float:
        """Calculate shortest distance between two line segments"""
        # Check if time intervals overlap

        if not (self.end_time >= other.start_time and self.start_time <= other.end_time):
            return float('inf')
        # Get overlapping time interval
        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)

        # Calculate CPA (Closest Point of Approach) time
        # Positions and velocities for CPA calculation
        p1 = np.array([self.start_point.x, self.start_point.y])
        p2 = np.array([other.start_point.x, other.start_point.y])

        v1 = np.array([(self.end_point.x - self.start_point.x) / (self.end_time - self.start_time),
                       (self.end_point.y - self.start_point.y) / (self.end_time - self.start_time)])
        v2 = np.array([(other.end_point.x - other.start_point.x) / (other.end_time - other.start_time),
                       (other.end_point.y - other.start_point.y) / (other.end_time - other.start_time)])
        # Relative velocity
        v_rel = v1 - v2
        # Time since overlap_start
        t0 = overlap_start
        p1_t0 = p1 + v1 * (t0 - self.end_time)
        p2_t0 = p2 + v2 * (t0 - other.start_time)

        # CPA calculation
        if np.linalg.norm(v_rel) < 1e-6:
            # Objects moving parallel - distance is constant
            return np.linalg.norm(p1_t0 - p2_t0)

        # Time of CPA within overlap interval
        t_cpa = t0 - np.dot(p1_t0 - p2_t0, v_rel) / np.dot(v_rel, v_rel)

        # Clamp to overlap interval
        t_cpa = max(overlap_start, min(overlap_end, t_cpa))

        # Positions at CPA time
        p1_cpa = p1 + v1 * (t_cpa - self.start_time)
        p2_cpa = p2 + v2 * (t_cpa - other.start_time)

        return np.linalg.norm(p1_cpa - p2_cpa)


# ================================
# 9. SimplifiedTrajectory (简化后的轨迹数据表达结构)
# ================================
@dataclass
class SimplifiedTrajectory:
    """Represents a simplified version of a trajectory using line segments"""
    oid: int
    segments: List[LineSegment]
    global_tolerance: float

    def __post_init__(self):
        self.segments.sort(key=lambda s: s.start_time)
        self.start_time = self.segments[0].start_time
        self.end_time = self.segments[-1].end_time

    def get_segment_at_time(self, t: int) -> Optional[LineSegment]:
        """Get the segment that covers time t"""
        for seg in self.segments:
            if seg.start_time <= t <= seg.end_time:
                return seg
        return None


# ================================
# 9. TrajectoryLoader (CSV 轨迹文件读取器)
# ================================
class TrajectoryLoader:
    @staticmethod
    def load_from_csv(filepath: str) -> Dict[str, MovingObject]:
        objects: Dict[str, MovingObject] = {}
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                oid, x, y, t, truth = row[0], float(row[1]), float(row[2]), int(row[3]), int(row[4]),
                point = TrajectoryPoint(x, y, t, oid, truth)
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
                oid, x, y, t, truth = row[1].oid, float(row[1].geometry.x), float(row[1].geometry.y), int(row[1].t), \
                                      row[1].truth
            except:
                oid, x, y, t, truth = row[1].oid, float(row[1].geometry.x), float(row[1].geometry.y), int(row[1].time), \
                                      row[1].truth

            point = TrajectoryPoint(x, y, t, oid, truth)
            if oid not in objects:
                objects[oid] = MovingObject(oid)
            objects[oid].add_point(point)
        return objects


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
# 7. The most basic companion pattern class, which can be inherited by various other companion pattern classes (all data at a single timestamp)
# ================================
class BasicCompanion(object):
    def __init__(self, pattern_id=0, start_time=0, end_time=0):
        self.time_to_objects: Dict[int, Set[str]] = defaultdict(set)  # t -> set of objects
        self.object_to_times: Dict[str, Set[int]] = defaultdict(set)  # o -> set of times
        self.extended_at_times: Dict[int, int] = defaultdict(lambda: 0)
        self.persist_members: Set[str] = set()
        self.lifetime = 0
        self.pattern_id = pattern_id
        self.extended = False
        self.start_time = start_time
        self.end_time = end_time

    def get_time_set(self) -> Set[int]:
        return set(self.time_to_objects.keys())

    def get_object_set(self) -> Set[str]:
        return set(self.object_to_times.keys())
        # return set(self.object_to_times.keys())

    def extend_pattern(self, time: int, objects: Set[str]):
        self.time_to_objects[time].update(objects)
        for o in objects:
            self.object_to_times[o].add(time)
        self.lifetime = len(self.time_to_objects.keys())

    def extend_pattern_by_another_pattern(self, objects: Set[str], other: 'BasicCompanion'):
        for t in other.time_to_objects.keys():
            self.time_to_objects[t].update(other.time_to_objects[t])
            for o in other.time_to_objects[t]:
                self.object_to_times[o].add(t)
        self.end_time = other.end_time
        self.lifetime = len(self.time_to_objects.keys())
        self.persist_members = objects

    # @property
    # def lifetime(self) -> int:
    #     return len(self.time_to_objects)

    def __repr__(self):
        return f"Pattern(objects={len(self.get_object_set())}, times={len(self.get_time_set())})"


class MovingClusterPattern(BasicCompanion):
    def __init__(self):
        super().__init__()
        self.removed = False


# @dataclass
class ConvoyPattern(BasicCompanion):
    # extended = False
    # start_time: int = field(init=False)
    # end_time: int = field(init=False)
    # lifetime: float
    def __init__(self, objects: Set[str] = None, start_time=0, end_time=0):
        super().__init__()
        self.objects: Set[str] = objects
        # self.lifetime = 0

    # def extend_pattern(self, time: int, objects: Set[str]):
    #     self.objects = objects

    # @property
    # def lifetime(self):
    #     return self.end_time - self.start_time + 1


class MovingFlockPattern(BasicCompanion):
    def __init__(self):
        super().__init__()


class FlockPattern(BasicCompanion):
    def __init__(self, radius=0):
        super().__init__()
        self.persist_members: Set[str]  # constant objects in the flock
        self.radius: float = radius  # radius of the containing disk
        self.spatial_extent: float = 0.0
        self.pattern_status = 'moving'  # moving or stationary
        self.base_id = None  # used for prune redundant flocks in IJGIS 2011

    def init_flock(self, objects: Set[str], start_time: int, end_time: int):
        self.persist_members = objects
        self.start_time = start_time
        self.end_time = end_time

    def set_base_obj(self, base_id: str):
        self.base_id = base_id

    def update_flock(self, objects: Set[str], end_time: int):
        self.persist_members = objects
        self.end_time = end_time

    def extend_pattern_multiple_times(self, objects: Set[str], start_time: int, end_time: int):

        for t in range(start_time, end_time + 1):
            self.time_to_objects[t].update(objects)
            for o in objects:
                self.object_to_times[o].add(t)
        self.lifetime = len(self.time_to_objects.keys())

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time


@dataclass
class Disc:
    """Represents a disc in space-time"""
    center: TrajectoryPoint
    radius: float
    object_ids: Set[str]


@dataclass
class Circle:
    """Represent a circle in 2D space"""
    center: TrajectoryPoint
    radius: float


class SwarmTreeNode:
    """Represents a node in the swarm search tree"""

    def __init__(self, objectset: Set[str], max_timeset: Set[int], confidence: float = 1.0):
        self.objectset = objectset  # set of object IDs at this node
        self.max_timeset = max_timeset  # maximal timeset for this objectset at this node
        self.confidence = confidence  # confidence score for probabilistic data
        self.children = []  # child nodes

    def add_child(self, child_node: 'SwarmTreeNode'):
        """Add a child node to this node"""
        self.children.append(child_node)

    def __repr__(self):
        return f"SwarmNode(objects={len(self.objectset)}, times={len(self.max_timeset)})"


class SwarmPattern(BasicCompanion):
    def __init__(self, objectset: Set[str], max_timeset: Set[str]):
        super().__init__()
        self.max_timeset = max_timeset
        self.objectset = objectset

    def extend_swarm_multiple_times(self, objectset: Set[str], max_timeset: Set[int]):

        for t in max_timeset:
            self.time_to_objects[t].update(objectset)
            for o in objectset:
                self.object_to_times[o].add(t)
        self.lifetime = len(self.time_to_objects.keys())


class TravelingCompanion(BasicCompanion):
    def __init__(self):
        super().__init__()

    def size(self) -> int:
        return len(self.object_to_times.keys())

    def is_companion(self, min_s: int, min_duration: int) -> bool:
        """Check if this candidate qualifies as a traveling companion"""
        return (self.size() >= min_s and
                self.lifetime >= min_duration)


@dataclass
class TravelingCompanionCandidate:
    """Represents a potential traveling companion"""
    objects: Set[str]  # object IDs
    duration: int  # how many snapshots this candidate has existed
    first_seen: int  # first snapshot where this candidate appeared
    is_closed: bool = False
    timestamps: List[int] = None

    def __post_init__(self):
        self.timestamps: List[int] = [self.first_seen]

    def size(self) -> int:
        return len(self.objects)

    def is_companion(self, min_s: int, min_duration: int) -> bool:
        """Check if this candidate qualifies as a traveling companion"""
        return (self.size() >= min_s and
                self.duration >= min_duration)


@dataclass
class TravelingBuddy:
    """Represent a micro-group of tightly bound objects"""
    buddy_id: int
    objects: Set[str]  # Object ids
    center: TrajectoryPoint
    radius: float  # maximum distance from center to any member in the buddy
    r_pt: TrajectoryPoint = None
    candidate_ids: Set[int] = None  # IDs of candidates containing this buddy

    def __post_init__(self):
        if self.candidate_ids is None:
            self.candidate_ids = set()

    def update_center(self, new_positions: Dict[str, TrajectoryPoint]):
        if not self.objects:
            return

        sum_x = 0
        sum_y = 0
        count = 0
        new_objs = []
        for obj_id in self.objects:
            if obj_id in new_positions.keys():
                new_objs.append(obj_id)
                sum_x += new_positions[obj_id].x
                sum_y += new_positions[obj_id].y
                count += 1
        if count >= 0:
            self.center = TrajectoryPoint(x=sum_x / count, y=sum_y / count, time=new_positions[obj_id].time)
            # Recalculate radius of the buddy
            max_dist = 0
            r_pt = None
            for obj_id in new_objs:
                dist = math.sqrt(
                    (new_positions[obj_id].x - self.center.x) ** 2 + (new_positions[obj_id].y - self.center.y) ** 2)
                if dist > max_dist:
                    max_dist = dist
                    r_pt = new_positions[obj_id]
            self.radius = max_dist
            self.r_pt = r_pt


@dataclass
class Crowd:
    """Represent a crowd - a sequence of consecutive snapshot clusters"""
    clusters: List[ClusterClass]  # time -> Cluster, consecutive clusters forming the crowd
    participators: Optional[Set[str]] = None  # objects that appear in at least k_p clusters

    @property
    def lifetime(self) -> int:
        return len(self.clusters)

    @property
    def start_time(self) -> int:
        if len(self.clusters) > 0:
            return self.clusters[0].time
        else:
            raise ValueError('The crowd is empty')

    @property
    def end_time(self) -> int:
        if len(self.clusters) > 0:
            return self.clusters[-1].time
        else:
            raise ValueError('The crowd is empty')

    def extend_crowd(self, cluster: ClusterClass):
        self.clusters.append(cluster)

    def is_valid(self, k_c: int, ) -> bool:
        """
        Check if the crowd meets the minimum member, lifetime, and cluster distance requirements
        Since we have ensure the cluster member size and cluster distance when adding a new cluster to the crowd,
        here we only need to check the lifetime constraint.
        """
        return self.lifetime >= k_c

    def find_participators(self, k_p: int) -> Set[str]:
        """Find all objects that appear in at least k_p clusters"""
        object_counts = defaultdict(int)
        for cluster in self.clusters:
            for obj_id in cluster.members:
                object_counts[obj_id] += 1

        self.participators = {obj_id for obj_id, count in object_counts.items() if count >= k_p}
        return self.participators

    def is_gathering(self, m_p: int, k_p: int) -> bool:
        """Check if this crowd is a gathering (has enough participators in each cluster)"""
        if self.participators is None:
            self.find_participators(k_p)

        # Check each cluster has at m_p participators
        for cluster in self.clusters:
            participators_in_cluster = len(cluster.members & self.participators)
            if participators_in_cluster < m_p:
                return False
        return True


@dataclass
class Gathering(Crowd):
    """Represents a gathering pattern. It in fact is a special kind of crowd"""
    pass


class Platoon(BasicCompanion):
    """A platoon pattern consisting objects and timestamps"""

    def __init__(self, objectset: Tuple, timestamps: List[int], N: int = 0):
        super().__init__()
        self.timestamps = timestamps
        self.objectset = objectset
        self.N = N

    def is_closed(self, other: 'Platoon') -> bool:
        """Check if this platoon is closed with respect to another platoon"""
        # A platoon is closed if it's both object-maximal and time-maximal
        if set(self.objectset).issubset(other.objectset) and (set(self.timestamps) == set(other.timestamps)):
            return False

        if self.objectset == other.objectset and (set(self.timestamps).issubset(set(other.timestamps))):
            return False

        return True

    def __repr__(self):
        return f'Platoon (objects={self.objectset}, timestamps={self.timestamps})'


class PrefixListEntry:
    """
     In PLo (prefix list) we store for a prefix P:
      - Tp : timestamp sequence (could have duplicates)
      - Np : number of occurrences (count of Tp entries)
    """

    def __init__(self):
        self.Tp = []  # list of timestamps (with duplicates allowed)
        self.Np = 0


class PTEntry:
    """
    Entry for an object 'o' in a PrefixTable PTX
    - Tmax: merged timestamps where objectset {o} ∪ X occurs (with duplicates)
    - Ncon: occurrences of locally-consecutive timestamp segments (after Extract-LC)
    - PLo: second-level hash: mapping prefix P -> PrefixListEntry
    """

    def __init__(self):
        self.Tmax = []  # list of timestamps (duplicates allowed)
        self.Ncon = 0
        self.Sminc_con = []  # list of maximally consecutive timestamp segments (each segment is list)
        self.PLo: Dict[Tuple, PrefixListEntry] = dict()


class PrefixTable:
    """
     PrefixTable associated with suffix X (PTX).
       - table: mapping object o -> PTEntry
       - order: list of objects in lexicographic (or chosen) order (used to process in reversed order)
    """

    def __init__(self, suffix_X: Tuple):
        # 主表结构: object -> {'T_max': [], 'N': int, 'prefixes': PrefixTable}
        self.suffix = tuple(suffix_X)
        self.table: Dict = dict()
        self.order: List = []

    def _create_entry(self):
        return {
            'T_max': [],
            'N': 0,
            'prefixes': defaultdict(lambda: {'T': [], 'N': 0})  # 递归定义, 支持多层次前缀
        }

    def add_cluster(self, objects: Set[str], timestamp: int):
        """添加一个聚类观测到前缀表中"""
        sorted_objects = sorted(objects)  # 保持固定顺序
        for i, obj in enumerate(sorted_objects):
            entry = self.table[obj]
            entry['T_max'].append(timestamp)
            entry['N'] += 1

            # 添加前缀信息（当前对象之前的所有对象）

            for prefix_size in range(1, i + 1):
                for prefix in itertools.combinations(sorted_objects[:i], prefix_size):
                    prefix_set = frozenset(prefix)
                    entry['prefixes'][prefix_set]['T'].append(timestamp)
                    entry['prefixes'][prefix_set]['N'] += 1

            # if i > 0:
            #     prefix_objects = frozenset(sorted_objects[:i])
            #     entry['prefixes'].add_cluster(set(prefix_objects), timestamp)

    def get_common_prefixes(self, min_t: int) -> Dict[FrozenSet[str], Dict]:
        """找出所有满足最小支持度的共同前缀组合"""

        common_prefixes = {}

        for obj, data in self.table.items():
            if data['N'] >= min_t:
                # 检查当前对象的前缀表
                prefix_results = data['prefixes'].get_common_prefixes(min_t)

                for prefix_set, prefix_data in prefix_results.items():
                    combined_set = prefix_set.union({obj})
                    common_prefixes[combined_set] = {
                        'T_max': list(set(data['T_max']) & set(prefix_data['T_max'])),
                        'N': prefix_data['N']
                    }
        return common_prefixes


@dataclass
class ClusterSet:
    """A cluster-set at a single time slot: a set/list of clusters whose union == OG (pattern members).

    Attributes
    ----------
    t : int
        Time slot index.
    clusters : Tuple[Cluster, ...]
        The constituent clusters for this slot.
    union_members : frozenset[str]
        Convenience precomputed union for quick checks
    """
    t: int
    clusters: Tuple[ClusterClass, ...]
    union_members: Set[str]

    @staticmethod
    def from_clusters(t: int, clusters: Iterable[ClusterClass]) -> 'ClusterSet':
        cluster_list = tuple(clusters)
        union = set(cluster_list[0].members)
        for c in cluster_list[1:]:
            union.union(c.members)
        return ClusterSet(t=t, clusters=cluster_list, union_members=union)

    def is_full_gathering(self) -> bool:
        """True if the cluster-set is a single cluster (i.e., all members are together)"""
        return len(self.clusters) == 1


class LTCPPattern(BasicCompanion):
    """""A Loose Traveling Companion Pattern"""

    def __init__(self, object_group: Set[str], cluster_sequences: List[ClusterSet], valid: bool = False):
        super().__init__()
        self.object_group = object_group
        self.TS = cluster_sequences  # ordered by time
        self.TS_dict: Dict[int, ClusterSet] = {}
        self.OG: Set[str] = object_group
        self.valid: bool = valid
        self.timestamps = [c.t for c in cluster_sequences]

    def extend_LTCP(self, t: int, cs: ClusterSet, OG: Set[str]):
        self.TS = self.TS + [cs]
        self.TS_dict[t] = cs
        self.OG = OG

    def start_time(self):
        return self.TS[0].t if self.TS else -1

    def get_end_time(self):
        return self.TS[-1].t if self.TS else -1

    def get_lifetime(self) -> int:
        return len(self.TS)

    def freq_full_gathering(self):
        """Count time slots where all members gather in exactly one cluster whose members == OG."""
        f = 0
        for cs in self.TS:
            if cs.is_full_gathering() and cs.union_members == self.OG:
                f += 1
        return f

    def __repr__(self):
        return f"Pattern(objects={len(self.OG)}, times={len(self.TS)})"


@dataclass
class MicroGroup:
    """ Represents a micro-group of moving objects as defined in the paper"""
    rep_id: str  # ID of representative object
    member_ids: Set[str]  # IDs of member objects
    radius: float  # Radius of the micro-group

    def size(self) -> int:
        """Total number of objects in this micro-group"""
        return 1 + len(self.member_ids)

    def get_all_members(self) -> Set[str]:
        """Get all object IDs in this micro-group including representative"""
        return {self.rep_id} | self.member_ids

    def _compute_radius(self):
        self.radius = 0


@dataclass
class LooseGroupCandidate:
    """Represents a candidate for loose group companion"""
    member_ids: Set[str]  # Object IDs in this candidate
    duration: int  # How long this candidate has existed
    leave_times: Dict[str, int]  # How long each member has been away
    extended: bool = False
    start_time: int = -1
    end_time: int = -1

    def size(self) -> int:
        """Number of members in this candidate"""
        return len(self.member_ids)


@dataclass
class LooseGroupCompanion:
    """Represents a discovered loose group companion"""
    member_ids: Set[str]  # Object IDs in this companion group
    duration: int  # Duration this group has been together
    timestamps: List[int] = None

    def update_time(self, t: int):
        self.timestamps.append(t)
        self.duration += 1


@dataclass
class DynamicConvoy:
    """A dynamic convoy with persistent and dynamic members"""
    persistent_members: Set[str]
    dynamic_members: Set[str]
    start_time: int
    end_time: int
    time_to_objects: Dict[int, List[str]] = None
    extended: bool = False

    def all_members(self) -> Set[str]:
        """Get all members of the dynamic convoy (persistent + dynamic)"""
        return self.persistent_members.union(self.dynamic_members)

    def duration(self) -> int:
        """Get duration of the convoy"""
        return self.end_time - self.start_time

    def is_w_convoy(self, w: int) -> bool:
        """Check if this convoy is a w-convoy (duration exactly w)"""
        return self.duration() == w

    def has_common_persistent_members(self, other: 'DynamicConvoy', m: int) -> bool:
        """Check if two dynamic convoys share at least m persistent members"""
        return len(self.persistent_members & other.persistent_members) >= m


@dataclass
class EvolvingConvoyStage:
    """A stage in an evolving convoy's lifecycle"""
    members: Set[str]
    start_time: int
    end_time: int
    convoy: DynamicConvoy = None
    extended: bool = False


@dataclass
class EvolvingConvoy:
    """An evolving convoy consisting of multiple stages"""
    stages: List[DynamicConvoy]
    start_time: int
    end_time: int

    def current_members(self, t: int) -> Set[str]:
        """Get members of the current stage"""
        return self.stages[-1].time_to_objects.get(t) if self.stages else set()

    def start_time(self) -> int:
        """Get start time of the first stage"""
        return self.stages[0].start_time if self.stages else 0

    def end_time(self) -> int:
        """Get end time of the first stage"""
        return self.stages[-1].start_time if self.stages else 0

    def duration(self) -> int:
        """Get total duration of the evolving convoy"""
        return self.end_time() - self.start_time() + 1

    def add_stage(self, members: Set[str], start_time: int, end_time: int):
        """Add a new stage to the evolving convoy"""
        self.stages.append(
            DynamicConvoy(persistent_members=members, dynamic_members=set(), start_time=start_time, end_time=end_time))


@dataclass
class CoMovementPattern:
    """A general co-movement pattern, which can represent group, flock, convoy, swarm, and platoon (constant members)"""
    object_ids: Set[str]  # set of object ids
    timestamps: List[int]
    time_to_objects: List[Tuple[int, List[str]]]
    type: str = "GCMP"  # Can be flock, convoy, swarm, platoon, etc.

    def duration(self) -> int:
        """Duration of the pattern in number of timestamps"""
        return len(self.timestamps)

    def size(self) -> int:
        """"Number of objects in the pattern"""
        return len(self.object_ids)

    def __repr__(self) -> str:
        return f"GCMP(objects={sorted(self.object_ids)}, timestamps={self.timestamps})"




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



