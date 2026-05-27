#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _flock.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/7/27 18:50
import copy
from collections import defaultdict
import numpy as np
from moveminerx.mining._basic_class import MovingObject, Snapshot, SnapshotBuilder, TrajectoryPoint, FlockPattern, Circle
from typing import Dict, Set


class MovingFlockMiner:
    """Implements the moving flock pattern detection algorithm"""

    def __init__(self, objects: Dict[str, MovingObject], m=2, radius=10, k=2, sync_rate=None, method='init',
                 metric='precomputed'):
        """
        Complete moving flock detection pipeline.
        :param objects: objects
        :param radius: Maximum distance for objects to be considered neighbors
        :param m:  Minimum number of objects in a flock
        :param radius: Maximum distance for objects to be considered neighbors
        :param k: Minimum duration (in time slices) for a flock
        :param sync_rate: Time interval between synchronized samples (in seconds)
        :param method: "init" for using the method in the original paper, "optimized" for using the optimized method

        :param metric: used to assign the method for similarity calculation
        Returns:
            List of detected moving flock patterns
        """

        self.m = m
        self.radius = radius
        self.k = k
        self.sync_rate = sync_rate
        self.metric = metric
        self.method = method
        self.objects = objects

    def run(self):
        pattern_counter = 0
        F_active_patterns: Dict[int, FlockPattern] = {}  # 当前活跃的flock序列，每个元素为 List[]
        F_final_patterns: Dict[int, FlockPattern] = {}  #
        # Step 1: Synchronize trajectories
        if self.sync_rate is None:
            snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(self.objects)
        else:
            snapshots = self._synchronize_trajectories(objects=self.objects, sync_rate=self.sync_rate)
        if len(snapshots) < 1:
            return []

        sorted_times = sorted(snapshots.keys())
        if self.method == 'optimized':
            for idx, t in enumerate(sorted_times):
                # step2: spatial neighbor computation that considers redundant
                snapshot = snapshots[t]
                neighbor = self._find_spatial_neighbors(snapshot=snapshot, radius=radius)
                # Step 3: Analyze membership persistence, it aims to check whether the members of the current basic disk
                F_next: Dict[int, FlockPattern] = {}
                for oid, neighbor in neighbor.items():
                    assigned = False
                    for fid, f in F_active_patterns.items():
                        f.extended = False
                        # last_f_t_objects are also the persist members of the flock at current timestamp
                        last_f_t_objects = f.time_to_objects[sorted_times[idx - 1]]
                        # Check if they share enough common members
                        inter = last_f_t_objects & neighbor

                        if len(inter) >= self.m:
                            f.extend_pattern(t, objects=inter)
                            f.update_flock(objects=inter, end_time=t)
                            f.extended = True
                            # update the next flock set
                            F_next[fid] = f
                            assigned = True

                    if not assigned:
                        # 新建一个moving flock
                        new_f = FlockPattern(radius=self.radius)
                        new_f.init_flock(objects=neighbor, start_time=t, end_time=t)
                        new_f.extend_pattern(t, neighbor)
                        # new_f.update_flock(objects=neighbor, end_time=t)
                        pattern_counter += 1
                        new_f.pattern_counter = pattern_counter
                        F_next[pattern_counter] = new_f

                for fid, f in F_active_patterns.items():
                    # 输出或者删除不能被扩展的 flock pattern
                    if not f.extended:
                        if f.duration >= self.k:
                            F_final_patterns[fid] = f
                F_active_patterns = F_next

            # 收尾: 剩余活跃模式判断是否满足条件
            for fid, f in F_active_patterns.items():
                if f.duration >= self.k:
                    F_final_patterns[fid] = f

            return F_final_patterns
        else:
            # method=='init'
            # Step 2: Find spatial neighbors
            neighbors = self._find_spatial_neighbors2(snapshots=snapshots, radius=self.radius)

            # Step 3: Analyze membership persistence
            candidate_flocks = self._analyze_membership_persistence(
                neighbors, self.m, self.k, radius=self.radius)

            # Step 4a: Prune redundant flocks according the overlap ratios of objects and timestamps
            non_redundant = self._prune_redundant_flocks(candidate_flocks)
            # non_redundant = self._prune_redundant_flocks2(candidate_flocks)
            # Step 4b: Prune stationary flocks
            moving_flocks = self._prune_stationary_flocks(non_redundant, snapshots, self.radius)
            # print('moving_flocks', moving_flocks)
            return moving_flocks

    def _synchronize_trajectories(self, objects: Dict[str, 'MovingObject'], sync_rate: float) -> Dict[int, 'Snapshot']:
        """
        Step 1: Synchronize trajectories by sampling points at regular intervals.
        :param objects:
        :param sync_rate: Time interval between synchronized samples (in seconds)
        Returns:
            Dict of Snapshots containing synchronized points for all objects
        """
        if len(objects) < 1:
            return {}

        # Find the overall time range
        min_max_times = [[obj.trajectory.start_time, obj.trajectory.end_time] for oid, obj in objects.items()]
        min_time = min(np.array(min_max_times)[:, 0])
        max_time = max(np.array(min_max_times)[:, 1])
        # Generate synchronized timestamps
        sync_times = np.arange(min_time, max_time + sync_rate, sync_rate)
        # Create snapshots at each sync time
        snapshots: Dict[int, 'Snapshot'] = {}
        for time in sync_times:
            points: Dict[str, 'TrajectoryPoint'] = {}
            for oid, obj in objects.items():
                point = obj.trajectory.interpolate_point(time)
                if point:
                    points[oid] = point  # Only add snapshot if there are points
            if len(points) > 0:
                snapshot = Snapshot(time=time)
                snapshot.points = points
                snapshots[time] = snapshot

        return snapshots

    def _find_spatial_neighbors(self, snapshot: Snapshot, radius: float) -> Dict[str, Set[str]]:
        """
        Step 2: Find spatial neighbors for each object at each time instance.
        :param snapshot: synchronized snapshot at certain time
        :param radius:  Maximum distance for objects to be considered neighbors
        :return:
            Dictionary mapping object_id to list of neighbor sets (one per snapshot)
        """
        neighbors: [str, Set] = {}  # {oid: set(neighbors), oid2: ....}
        used_objs = set()
        for oid, pt in snapshot.points.items():
            # Get all neighbors (including self)
            # neighbor_set = snapshot.get_neighbors(pt, radius)
            new_neighbor_set = set()
            for oid_ in snapshot.get_neighbors(pt, radius):
                if oid_ not in used_objs:
                    new_neighbor_set.add(oid_)
                    used_objs.add(oid_)
            if len(new_neighbor_set) < 2:
                continue
            neighbors[oid] = new_neighbor_set

        return neighbors

    def _find_spatial_neighbors2(self, snapshots: Dict[int, Snapshot], radius: float) -> Dict[str, Dict[int, Set[str]]]:
        """
        Step 2: Find spatial neighbors for each object at each time instance.

        Parameters:
            snapshots: List of synchronized snapshots
            radius: Maximum distance for objects to be considered neighbors

        Returns:
            Dictionary mapping object_id to list of neighbor sets (one per snapshot)
        """
        neighbors = defaultdict(dict)

        for t, snapshot in snapshots.items():
            # used_objs = set()
            # new_neighbor_set = set()
            for obj_id, point in snapshot.points.items():
                # Get all neighbors (including self)
                neighbor_set = snapshot.get_neighbors(point, radius)
                # for oid_ in neighbor_set:
                #     if oid_ not in used_objs:
                #         new_neighbor_set.add(oid_)
                #         used_objs.add(oid_)
                # if len(new_neighbor_set) < 2:
                #     continue
                neighbors[obj_id][t] = neighbor_set

        return neighbors

    def _analyze_membership_persistence(self, neighbors: Dict[str, Dict[int, Set[str]]],
                                        m: int, k: int, radius: float) -> Dict[int, FlockPattern]:
        """
        Step 3: Analyze membership persistence to find flocks that last multiple time steps.
        Parameters:
            neighbors: Spatial neighbors from previous step
            m: Minimum number of objects in a flock
            k: Minimum duration (in time slices) for a flock
        Returns:
            Dict of candidate Flock patterns
        """
        flocks: Dict[int, FlockPattern] = {}
        pattern_id = 0
        # For each object as base trajectory
        for base_id, neighbor_sets in neighbors.items():
            if not neighbor_sets:
                continue
            # Initialize candidate flocks (one for each time slice)
            candidates = []
            for t, neighbor_set in neighbor_sets.items():
                if len(neighbor_set) >= m:
                    f = FlockPattern(radius=radius)
                    f.init_flock(objects=neighbor_set, start_time=t, end_time=t)
                    f.set_base_obj(base_id)
                    f.extend_pattern(t, neighbor_set)
                    candidates.append(f)
            # merge adjacent candidate flocks
            changed = True
            while changed:
                # print('base_id', base_id, 'length candidates', len(candidates))
                changed = False
                new_candidates = []
                i = 0
                # 从t0到t
                while i < len(candidates):
                    if i + 1 < len(candidates) and candidates[i].end_time + 1 == candidates[i + 1].start_time:
                        # Check if they share enough common members
                        inter = candidates[i].persist_members & candidates[i + 1].persist_members
                        if len(inter) >= m:
                            # Merge the two flocks
                            # merged_flock = FlockPattern(radius=radius)
                            # 将两个flocks合并 flock1: candidate[i], flock2: candidate[i+1]
                            merged_flock = copy.deepcopy(candidates[i])
                            # merged_flock.init_flock(objects=inter, start_time=candidates[i].start_time,
                            #                         end_time=candidates[i+1].end_time)
                            merged_flock.extend_pattern_by_another_pattern(objects=inter, other=candidates[i + 1])
                            # merged_flock.extend_pattern(candidates[i+1].end_time, objects=inter)

                            new_candidates.append(merged_flock)
                            i += 2  # Skip next since we merged it
                            changed = True
                            continue
                    new_candidates.append(candidates[i])
                    i += 1
                candidates = new_candidates
            # Keep only flocks that meet the minimum duration
            for flock in candidates:
                if flock.duration >= k:
                    pattern_id += 1
                    flock.pattern_id = pattern_id
                    flocks[pattern_id] = flock

        return flocks

    def _prune_stationary_flocks(self, flocks: Dict[int, FlockPattern], snapshots: Dict[int, Snapshot], radius=5) -> \
            Dict[int, FlockPattern]:
        """
        Step 4b: Prune stationary flocks based on spatial extent.
        :param flocks: Dict of discovered flocks
        :param snapshots: Dict of synchronized snapshots
        :param radius:
        Returns:
            Dict of moving flocks (stationary ones removed)
        """
        moving_flocks: Dict[int, FlockPattern] = {}

        for fid, flock in flocks.items():
            # Compute spatial extent of this flock
            extent = self._compute_spatial_extent(flock=flock, snapshots=snapshots)
            flock.spatial_extent = extent
            # keep only flocks with extent >= radius
            if extent >= radius:
                moving_flocks[fid] = flock
            else:
                flock.pattern_status = 'stationary'

        return moving_flocks

    def _compute_spatial_extent(self, flock: FlockPattern, snapshots: Dict[int, Snapshot]) -> float:
        """
        Compute the spatial extent of a flock during its time interval.

        Parameters
            flock: The flock to analyze
            snapshots: List of all synchronized snapshots

        Returns:
            The spatial extent (maximum of length/width of MBR)
        """

        if len(snapshots) < 1 or len(flock.persist_members) < 1:
            return 0.0

        # Get all points for flock members during the flock's time interval
        points = []

        for t in range(flock.start_time, flock.end_time + 1):

            snapshot = snapshots[t]
            for oid in flock.persist_members:
                if oid in snapshot.points.keys():
                    points.append(snapshot.points[oid])

        if len(points) < 1:
            return 0.0

        # Compute Minimum Bounding Rectangle (MRB)
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        length = max_x - min_x
        width = max_y - min_y

        return max(length, width)

    def _prune_redundant_flocks2(self, flocks: Dict[int, FlockPattern]) -> Dict[int, FlockPattern]:
        """
        Step 4a: Prune redundant flocks where one is a subset of another.

        Parameters
            flocks: Dict of candidate flocks

        Returns
            Dict of non-redundant flocks
        """
        # Sort flocks by size (descending) and duration (descending)
        sorted_flocks = sorted(list(flocks.values()), key=lambda f: (-len(f.persist_members), -f.duration))
        # print('sorted_flocks', sorted_flocks)
        non_redundant: Dict[int, FlockPattern] = {}
        seen_members = set()
        used_ = set()
        for idx, f in enumerate(sorted_flocks):

            # Create a unique identifier for this flock's members and time interval
            fid = (frozenset(f.persist_members), f.start_time, f.end_time)
            if fid in seen_members:
                continue
            # Check if this flock is similar to any already kept flock
            is_redundant = False

            for kept_flock in non_redundant.values():

                # Calculate overlap in members
                member_overlap = len(f.persist_members & kept_flock.persist_members)
                member_overlap_ratio = member_overlap / min(len(f.persist_members), len(kept_flock.persist_members))

                # Calculate temporal overlap
                time_overlap = min(f.end_time, kept_flock.end_time) - max(f.start_time, kept_flock.start_time) + 1
                time_overlap_ratio = time_overlap / min(f.duration, kept_flock.duration)

                # Consider redundant  if both overlaps are significant
                if member_overlap_ratio > 0.65 and time_overlap_ratio > 0.65:
                    is_redundant = True
                    break

            if not is_redundant:
                non_redundant[idx] = f
                seen_members.add(fid)

        return non_redundant

    def _prune_redundant_flocks(self, flocks: Dict[int, FlockPattern]) -> Dict[int, FlockPattern]:
        """
        Step 4a: Prune redundant flocks where one is a subset of another.

        Parameters
            flocks: Dict of candidate flocks

        Returns
            Dict of non-redundant flocks
        """
        # Sort flocks by size (descending) and duration (descending)
        sorted_flocks = sorted(list(flocks.values()), key=lambda f: (-len(f.persist_members), -f.duration))
        # print('sorted_flocks', sorted_flocks)
        non_redundant: Dict[int, FlockPattern] = {}
        seen_members = set()
        for idx, f in enumerate(sorted_flocks):
            # Create a unique identifier for this flock's members and time interval
            fid = (frozenset(f.persist_members), f.start_time, f.end_time)

            if fid in seen_members:
                continue
            # Check if this flock is similar to any already kept flock
            is_redundant = False
            # print(fid)
            for kept_flock in non_redundant.values():
                # Calculate overlap in members
                member_overlap = len(f.persist_members & kept_flock.persist_members)
                member_overlap_ratio = member_overlap / min(len(f.persist_members), len(kept_flock.persist_members))

                # Calculate temporal overlap
                time_overlap = min(f.end_time, kept_flock.end_time) - max(f.start_time, kept_flock.start_time) + 1
                time_overlap_ratio = time_overlap / min(f.duration, kept_flock.duration)

                # Consider redundant  if both overlaps are significant
                if member_overlap_ratio > 0.65 and time_overlap_ratio > 0.65:
                    is_redundant = True
                    break

            if not is_redundant:
                non_redundant[idx] = f
                seen_members.add(fid)

        return non_redundant
