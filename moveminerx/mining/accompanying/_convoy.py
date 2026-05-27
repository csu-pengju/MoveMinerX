#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _convoy.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/7/26 17:42

import math
import random
import numpy as np
from sklearn.cluster import DBSCAN
from moveminerx.mining._basic_class import MovingObject, Snapshot, SnapshotBuilder, ConvoyPattern, ClusterClass, \
    LineSegment, Trajectory, SimplifiedTrajectory, TrajectoryPoint
from typing import List, Tuple, Dict


class CMCAlgorithm:
    """
    CMC (Coherent Moving Clusters) Algorithm Implementation
    Discovery of Convoys in Trajectory Databases (Jeung et al. 2008)
    """

    def __init__(self, objects: Dict[str, MovingObject], k: int = 2, m: int = 2, eps: float = 5,
                 min_pts: int = 2, metric: str = 'precomputed'):

        """
        The python completion of Discovery of Convoys in Trajectory Databases, Jeung et al. 2008.
        Convoy pattern detection.
        :param objects:
        :param k: the least timestamps of a valid convoy.
        :param m: the minimum constant number of a convoy.
        :param eps: the radius threshold used for DBSCAN used for moving object clustering at the timestamp.
        :param min_pts: the minimum neighbors(moving objects) of a moving object to be a core object of DBSCAN.
        :return:
        """

        self.objects = objects
        self.k = k  # 最小时间持续长度
        self.m = m  #
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric

    def run(self) -> Dict[int, ConvoyPattern]:
        snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(self.objects)
        sorted_times = sorted(snapshots.keys())
        pattern_id = 0
        active_patterns: Dict[int, ConvoyPattern] = {}  # 当前活跃的移动簇序列，每个元素为 List[ClusterClass]
        final_patterns: Dict[int, ConvoyPattern] = {}  #
        for t_idx, t in enumerate(sorted_times):
            # 1️⃣ clustering at current timestamp
            clusters = snapshots[t].cluster_snapshot(snapshot=snapshots[t], eps=self.eps, minPts=self.min_pts,
                                                     metric=self.metric)
            next_patterns: Dict[int, ConvoyPattern] = {}
            # 提前转 set（性能优化）
            cluster_members = {
                cid: set(c.members) for cid, c in clusters.items()
            }
            # 标记
            cluster_assigned = {cid: False for cid in clusters}
            pattern_extended = {pid: False for pid in active_patterns}

            # 2️⃣ Expanding existing convoys
            for pid, pattern in active_patterns.items():
                last_objects = pattern.objects
                # print(pattern.object_to_times)
                for cid, members in cluster_members.items():
                    inter = last_objects & members
                    if len(inter) >= self.m:
                        pattern.objects = inter
                        pattern.extend_pattern(t, objects=inter)
                        next_patterns[pid] = pattern
                        cluster_assigned[cid] = True
                        pattern_extended[pid] = True

            # 3️⃣ 新建 convoy
            for cid, members in cluster_members.items():
                if not cluster_assigned[cid]:
                    # 新建一个moving cluster
                    pattern_id += 1
                    new_pattern = ConvoyPattern()
                    new_pattern.pattern_id = pattern_id
                    new_pattern.extend_pattern(t, members)
                    new_pattern.objects = members
                    next_patterns[pattern_id] = new_pattern

            # 4️⃣ 处理未扩展的 convoy
            for pid, pattern in active_patterns.items():
                # 输出或者删除不能被扩展的moving cluster
                if not pattern_extended[pid]:
                    if pattern.lifetime > self.k:
                        final_patterns[pid] = pattern

            # 更新
            active_patterns = next_patterns

        # 5️⃣收尾: 剩余活跃模式判断是否满足条件
        for pid, pattern in active_patterns.items():
            if pattern.lifetime > self.k:
                final_patterns[pid] = pattern
        return final_patterns


class CuTSAlgorithm:

    def __init__(self, objects: Dict[str, MovingObject], k: int = 2, m: int = 2, eps: float = 5,
                 min_pts: int = 2, e=None, metric: str = 'precomputed'):
        self.objects = objects
        self.k = k  # 最小时间持续长度
        self.m = m  #
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric
        if e is None:
            self.e = eps
        else:
            self.e = e
        self.simplified_trajs = []
        self.max_time = max(obj.trajectory.end_time for obj in self.objects.values())
        self.min_time = min(obj.trajectory.start_time for obj in self.objects.values())

    def run(self, ) -> Dict[int, ConvoyPattern]:
        """
        Discover convoys using the CuTS algorithm
        """
        # Step 1: Simplify trajectories
        simplified_trajs = simplify_trajectories(self.e, objects=self.objects)
        # Step 2: Filter step - find candidate convoys
        candidates = filter_step(self.m, self.k, self.e, objects=self.objects, simplified_trajs=simplified_trajs,
                                 min_time=self.min_time, max_time=self.max_time)
        # Step 3: Refinement step - verify candidates
        convoys = refinement_step(candidates,  objects=self.objects, m=self.m, k=self.k, e=self.e, min_pts=self.min_pts, metric=self.metric)

        return convoys


class CuTSPlusAlgorithm:
    """   CuTS+ algorithm with faster trajectory simplification """
    def __init__(self, objects: Dict[str, MovingObject], k: int = 2, m: int = 2, eps: float = 5,
                 min_pts: int = 2, e=None, metric: str = 'precomputed'):
        self.objects = objects
        self.k = k  # 最小时间持续长度
        self.m = m  #
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric
        if e is None:
            self.e = eps
        else:
            self.e = e
        self.simplified_trajs = []
        self.max_time = max(obj.trajectory.end_time for obj in self.objects.values())
        self.min_time = min(obj.trajectory.start_time for obj in self.objects.values())

    def run(self):

        # Step 1: Simplify trajectories using fast douglas peucker algorithm
        delta = compute_delta(self.e, objects=self.objects)
        simplified_trajs = []
        for oid, obj in self.objects.items():
            traj = obj.trajectory
            simplified = fast_douglas_peucker(traj, delta)
            simplified_trajs.append(simplified)
        # Steps 2 and 3 is the same as the CuTS algorithm
        # Step 2: Filter step - find candidate convoys
        candidates = filter_step(self.m, self.k, self.e, objects=self.objects, simplified_trajs=simplified_trajs,
                                 min_time=self.min_time, max_time=self.max_time)
        # Step 3: Refinement step - verify candidates
        convoys = refinement_step(candidates, objects=self.objects, m=self.m, k=self.k, e=self.e, min_pts=self.min_pts,
                                  metric=self.metric)

        return convoys


class CuTSStarAlgorithm:
    """ CuTS* algorithm with faster trajectory simplification """

    def __init__(self, objects: Dict[str, MovingObject], k: int = 2, m: int = 2, eps: float = 5,
                 min_pts: int = 2, e=None, metric: str = 'precomputed'):
        self.objects = objects
        self.k = k  # 最小时间持续长度
        self.m = m  #
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric
        if e is None:
            self.e = eps
        else:
            self.e = e
        self.simplified_trajs = []
        self.max_time = max(obj.trajectory.end_time for obj in self.objects.values())
        self.min_time = min(obj.trajectory.start_time for obj in self.objects.values())

    def run(self):

        # Step 1: Simplify trajectories using temporal-aware douglas peucker algorithm
        delta = compute_delta(self.e, objects=self.objects)
        simplified_trajs = []
        for oid, obj in self.objects.items():
            traj = obj.trajectory
            simplified = temporal_aware_douglas_peucker(traj, delta)
            simplified_trajs.append(simplified)
        # Steps 2 and 3 is the same as the CuTS algorithm
        # Step 2: Filter step - find candidate convoys
        candidates = filter_step(self.m, self.k, self.e, objects=self.objects, simplified_trajs=simplified_trajs,
                                 min_time=self.min_time, max_time=self.max_time)
        # Step 3: Refinement step - verify candidates
        convoys = refinement_step(candidates, objects=self.objects, m=self.m, k=self.k, e=self.e, min_pts=self.min_pts,
                                  metric=self.metric)
        return convoys


def filter_step(m: int, k: int, e: float, objects, simplified_trajs, min_time, max_time) -> Dict[int, ConvoyPattern]:
    """Filter step of CuTS algorithm"""
    active_patterns: Dict[int, ConvoyPattern] = {}  # 当前活跃的移动簇序列，每个元素为 List[Cluster]
    final_patterns: Dict[int, ConvoyPattern] = {}  #
    # Determine time partition length
    lambda_ = compute_lambda(k, objects=objects, simplified_trajs=simplified_trajs)
    pattern_id = 0
    # Process time partitions, 每次处理lambda_个时间
    for t_start in range(min_time, max_time + 1, lambda_):
        t_end = min(t_start + lambda_ - 1, max_time)

        # Get all simplified trajectory segments that intersect this time partition
        segments = []
        for st in simplified_trajs:
            for seg in st.segments:
                if seg.start_time <= t_end and seg.end_time >= t_start:
                    segments.append((st.oid, seg))

        if len(segments) < m:
            continue  # Not enough objects to form a convoy

        # perform density-based clustering on segments
        clusters = trajectory_segment_dbscan(segments, e, m)

        V_next: Dict[int, ConvoyPattern] = {}

        for vid, v in active_patterns.items():
            best_cluster_id = None
            max_intersection = 0

            # Find cluster with maximum intersection
            for cid, c in clusters.items():
                # v_last_objects = v.time_to_objects[t_st]
                inter = len(v.get_object_set() & set(c.members))
                if inter >= m and inter > max_intersection:
                    max_intersection = inter
                    best_cluster_id = cid

            if best_cluster_id is not None:
                updated_ids = v.get_object_set() & set(clusters[best_cluster_id].members)
                v.objects = updated_ids
                v.extend_pattern(t_start, updated_ids)
                v.end_time = t_end
                v.lifetime += lambda_
                v.extended = True
                clusters[best_cluster_id].assigned = True
                #  update the next moving cluster set
                V_next[vid] = v

            if not v.extended and v.lifetime >= k:
                final_patterns[vid] = v

        for cid, c in clusters.items():
            if not c.assigned:
                # 新建一个convoy
                mv = ConvoyPattern(start_time=t_start, end_time=t_end)
                mv.extend_pattern(t_start, set(c.members))
                pattern_id += 1
                mv.pattern_id = pattern_id
                mv.lifetime += lambda_
                V_next[pattern_id] = mv
        active_patterns = V_next

    # 收尾: 剩余活跃模式判断是否满足条件
    for vid, v in active_patterns.items():
        if v.lifetime > k:
            final_patterns[vid] = v

    return final_patterns


def refinement_step(candidate_convoys: Dict[int, ConvoyPattern], objects, m: int = 2,
                    k: int = 2, e: float = 10,  min_pts: int = 2, metric='precomputed') -> Dict[int, ConvoyPattern]:
    """Refinement step of CuTS algorithm"""

    candidate_objects: Dict[str, MovingObject] = {}

    for vid, v in candidate_convoys.items():
        # Get original object of candidate objects
        for oid in v.get_object_set():
            for obj in objects.values():
                if obj.oid == oid:
                    candidate_objects[oid] = obj
                    break

    # Apply CMC only on the candidate time interval
    cmc = CMCAlgorithm(objects=candidate_objects, k=k, m=m, eps=e, min_pts=min_pts, metric=metric)
    refined_res = cmc.run()
    # refined_res = CMC(objects=candidate_objects, m=m, k=k, eps=e, minPts=m, clustering_metric='precomputed')
    return refined_res


def compute_lambda(k, objects, simplified_trajs):
    """Compute time partition length (lambda)"""
    # Calculate average reduction ratio
    total_original = sum(len(obj.trajectory.points) for obj in objects.values())
    total_simplified = sum(len(st.segments) for st in simplified_trajs)
    reduction_ratio = total_simplified / total_original if total_original > 0 else 1.0

    # Calculate average trajectory duration
    avg_duration = sum(obj.trajectory.end_time - obj.trajectory.start_time + 1 for obj in objects.values())

    # Calculate time coverage density
    # self.min_time = min(t.start_time for t in trajectories)
    # self.max_time = max(t.end_time for t in trajectories)
    max_time = max(obj.trajectory.end_time for obj in objects.values())
    min_time = min(obj.trajectory.start_time for obj in objects.values())
    total_duration = max_time - min_time + 1
    density = avg_duration / total_duration if total_duration > 0 else 1.0

    # Compute lambda
    lambda_ = int(avg_duration * reduction_ratio * (1 - density) + (2 / total_duration))

    # Ensure lambda is at least 1 and not too large
    return max(1, min(lambda_, k))


def compute_delta(e: float, objects) -> float:
    """Compute tolerance value for trajectory simplification"""
    # Sample some trajectories to determine delta
    sample_size = min(10, len(objects))
    sample_objs = random.sample(objects.items(), sample_size)

    # Collect actual tolerances from DP with delta = 0
    tolerances = []

    for obj in sample_objs:
        tolerances.extend(douglas_peucker_tolerances(obj[1].trajectory, 0))

    # Sort tolerances and find largest gap
    tolerances.sort()
    max_gap = 0
    best_delta = 0
    for i in range(1, len(tolerances)):
        if tolerances[i] > e:
            break  # Only consider tolerances <= e

        gap = tolerances[i] - tolerances[i - 1]
        if gap > max_gap:
            max_gap = gap
            best_delta = tolerances[i - 1]

    return best_delta if best_delta > 0 else e / 2


def simplify_trajectories(e: float, objects):
    """Simplify trajectories using Douglas-Peucker algorithm"""
    # 优化delta的计算
    simplified_trajs = []
    delta = compute_delta(e, objects)
    for oid, obj in objects.items():
        traj = obj.trajectory
        simplified = douglas_peucker(traj, delta)
        simplified_trajs.append(simplified)

    return simplified_trajs


def douglas_peucker(traj: Trajectory, delta: float):
    """
    Douglas-Peucker algorithm for trajectory simplification
    :param traj: the trajectory to be simplified by douglas peucker algorithm.
    :param delta this distance threshold is computed by function 'computeDelta'
    """
    if len(traj.points) <= 2:
        actual_tol = 0 if len(traj.points) == 1 else perpendicular_distance(
            traj.points_list[1][1], traj.points_list[0][1], traj.points_list[-1][1])
        segment = LineSegment(start_point=traj.points_list[0][1], end_point=traj.points_list[-1][1],
                              actual_tolerance=actual_tol)
        return SimplifiedTrajectory(traj.oid, [segment], delta)

    # Find point with maximum distance
    max_dist = 0
    max_index = 0
    start_point = traj.points_list[0][1]
    end_point = traj.points_list[-1][1]

    for i in range(1, len(traj.points) - 1):
        dist = perpendicular_distance(traj.points_list[i][1], start_point, end_point)
        if dist > max_dist:
            max_dist = dist
            max_index = i

    if max_dist <= delta:
        # All points are within tolerance - simplify to single segment
        actual_tol = max_dist
        segment = LineSegment(start_point, end_point, actual_tol)
        return SimplifiedTrajectory(traj.oid, [segment], delta)
    else:
        # Recursively simplify sub-trajectories

        left_traj = Trajectory(oid=traj.oid, points=traj.points_list[:max_index + 1])
        right_traj = Trajectory(oid=traj.oid, points=traj.points_list[max_index:])
        left_simplified = douglas_peucker(left_traj, delta)
        right_simplified = douglas_peucker(right_traj, delta)
        # Combine results
        segments = left_simplified.segments + right_simplified.segments
        return SimplifiedTrajectory(traj.oid, segments, delta)


def perpendicular_distance(point: TrajectoryPoint, line_start: TrajectoryPoint,
                           line_end: TrajectoryPoint) -> float:
    """Calculate perpendicular distance from point to line segment"""
    if line_start.x == line_end.x and line_start.y == line_end.y:
        return math.sqrt((point.x - line_start.x) ** 2 + (point.y - line_start.y) ** 2)

    # Vector from line_start to line_end
    seg_vec = np.array([line_end.x - line_start.x, line_end.y - line_start.y])

    # Vector from line_start to point
    point_vec = np.array([point.x - line_start.x, point.y - line_start.y])

    seg_length = np.linalg.norm(seg_vec)

    # Projection of point_vec onto seg_vec
    projection = np.dot(point_vec, seg_vec) / seg_length

    if projection < 0:
        # Closest to line_start
        return math.sqrt((point.x - line_start.x) ** 2 + (point.y - line_start.y) ** 2)
    elif projection > seg_length:
        # Closest to line_end
        return math.sqrt((point.x - line_end.x) ** 2 + (point.y - line_end.y) ** 2)
    else:
        # Distance to line
        return abs(np.cross(seg_vec, point_vec)) / seg_length


def douglas_peucker_tolerances(self, traj: Trajectory, delta: float) -> List[float]:
    """Run DP algorithm and collect all actual tolerances"""
    if len(traj.points) <= 2:
        return []

    # Find point with maximum distance
    max_dist = 0
    max_index = 0
    start_point = traj.points_list[0][1]
    end_point = traj.points_list[-1][1]

    for i in range(1, len(traj.points) - 1):
        dist = self.perpendicular_distance(traj.points_list[i][1], start_point, end_point)
        if dist > max_dist:
            max_dist = dist
            max_index = i

    if max_dist <= delta:
        return [max_dist]
    else:
        # Recursively process sub-trajectories
        left_traj = Trajectory(oid=traj.oid, points=traj.points_list[:max_index + 1])
        right_traj = Trajectory(oid=traj.oid, points=traj.points_list[max_index:])
        left_tolerance = self.douglas_peucker_tolerances(left_traj, delta)
        right_tolerance = self.douglas_peucker_tolerances(right_traj, delta)
    return left_tolerance + [max_dist] + right_tolerance


def trajectory_segment_dbscan(segments: List[Tuple[int, LineSegment]], eps: float, minSeg: int) \
        -> Dict[int, ClusterClass]:
    """Density-based clustering for trajectory segments"""

    distance_matrix = np.zeros((len(segments), len(segments)))

    for i in range(len(segments)):
        seg1 = segments[i][1]
        for j in range(i + 1, len(segments)):
            seg2 = segments[j][1]
            dist = seg1.distance_to_segment(seg2)
            distance_matrix[i][j] = dist

    distance_matrix = distance_matrix + distance_matrix.T

    labels = DBSCAN(eps=eps, min_samples=minSeg, metric='precomputed').fit(distance_matrix).labels_

    clusters: Dict[int, ClusterClass] = {}
    for i, label in enumerate(labels):
        if label != -1:
            if label not in clusters.keys():
                # cluster = ClusterClass(cid=label, )
                cluster = ClusterClass(cid=label, t=segments[i][1].start_point.time,
                                       members=[segments[i][1].start_point.oid])
                clusters[label] = cluster
            else:
                clusters[label].add_member(segments[i][1].start_point.oid)

    return clusters


def fast_douglas_peucker(traj: Trajectory, delta: float) -> SimplifiedTrajectory:
    """
        Douglas-Peucker+ algorithm for faster trajectory simplification
        :param traj: the trajectory to be simplified by douglas peucker algorithm.
        :param delta this distance threshold is computed by function 'computeDelta'
    """
    if len(traj.points) <= 2:
        # Create a single segment for trajectories with 2 or fewer points
        actual_tol = 0 if len(traj.points_list) == 1 else \
            perpendicular_distance(
                traj.points_list[1][1], traj.points_list[0][1], traj.points_list[-1][1])
        segment = LineSegment(start_point=traj.points_list[0][1], end_point=traj.points_list[-1][1],
                              actual_tolerance=actual_tol)
        return SimplifiedTrajectory(traj.oid, [segment], delta)

    # Find point closest to middle that exceeds delta
    middle_index = len(traj.points_list) // 2
    best_index = None
    min_dist_to_middle = float('Inf')
    start_point = traj.points_list[0][1]
    end_point = traj.points_list[-1][1]

    for i in range(1, len(traj.points) - 1):
        dist = perpendicular_distance(traj.points_list[i][1], start_point, end_point)
        if dist > delta:
            # This point exceeds tolerance - check if closest to middle
            dist_to_middle = abs(i - middle_index)
            if dist_to_middle < min_dist_to_middle:
                min_dist_to_middle = dist_to_middle
                best_index = i
    if best_index is None:
        # All points are within tolerance - simplify to single segment
        max_dist = 0
        for i in range(1, len(traj.points_list) - 1):
            dist = perpendicular_distance(traj.points_list[i][1], start_point, end_point)
            if dist > max_dist:
                max_dist = dist
        segment = LineSegment(start_point, end_point, max_dist)
        return SimplifiedTrajectory(traj.oid, [segment], delta)

    else:
        # Recursively simplify sub-trajectories
        left_traj = Trajectory(oid=traj.oid, points=traj.points_list[:best_index + 1])
        right_traj = Trajectory(oid=traj.oid, points=traj.points_list[best_index:])
        left_simplified = fast_douglas_peucker(left_traj, delta)
        right_simplified = fast_douglas_peucker(right_traj, delta)
        # Combine results
        segments = left_simplified.segments + right_simplified.segments
        return SimplifiedTrajectory(traj.oid, segments, delta)


def temporal_aware_douglas_peucker(traj: Trajectory, delta: float) -> SimplifiedTrajectory:
    """
    Temporal-aware Douglas-Peucker algorithm (DP*)
    :param traj: the trajectory to be simplified by douglas peucker algorithm.
    :param delta this distance threshold is computed by function 'computeDelta'
    """
    if len(traj.points) <= 2:
        # Create a single segment for trajectories with 2 or fewer points
        actual_tol = 0 if len(traj.points_list) == 1 else \
            perpendicular_distance(
                traj.points_list[1][1], traj.points_list[0][1], traj.points_list[-1][1])
        segment = LineSegment(start_point=traj.points_list[0][1], end_point=traj.points_list[-1][1],
                              actual_tolerance=actual_tol)
        return SimplifiedTrajectory(traj.oid, [segment], delta)

    # Find point with maximum time-ratio distance
    max_dist = 0
    max_index = 0
    start_point = traj.points_list[0][1]
    end_point = traj.points_list[-1][1]

    for i in range(1, len(traj.points) - 1):
        dist = time_ratio_distance(traj.points_list[i][1], start_point, end_point)
        if dist > max_dist:
            max_dist = dist
            max_index = i

    if max_dist <= delta:
        # All points are within tolerance - simplify to single segment
        actual_tol = max_dist
        segment = LineSegment(start_point, end_point, actual_tol)
        return SimplifiedTrajectory(traj.oid, [segment], delta)

    else:
        # Recursively simplify sub-trajectories
        left_traj = Trajectory(oid=traj.oid, points=traj.points_list[:max_index + 1])
        right_traj = Trajectory(oid=traj.oid, points=traj.points_list[max_index:])
        left_simplified = temporal_aware_douglas_peucker(left_traj, delta)
        right_simplified = temporal_aware_douglas_peucker(right_traj, delta)
        # Combine results
        segments = left_simplified.segments + right_simplified.segments
        return SimplifiedTrajectory(traj.oid, segments, delta)


def time_ratio_distance(point: TrajectoryPoint, line_start: TrajectoryPoint,
                            line_end: TrajectoryPoint) -> float:
    """Calculate time-ratio distance (DP* distance)"""
    if line_start.time == line_end.time:
        return math.sqrt((point.x - line_start.x) ** 2 + (point.y - line_start.y) ** 2)

    # Calculate time ratio
    ratio = (point.time - line_start.time) / (line_end.time - line_start.time)

    # Calculate point on line at this time ratio
    line_x = line_start.x + ratio * (line_end.x - line_start.x)
    line_y = line_start.y + ratio * (line_end.y - line_start.y)

    # Euclidean distance between point and line point
    return math.sqrt((point.x - line_x) ** 2 + (point.y - line_y) ** 2)
