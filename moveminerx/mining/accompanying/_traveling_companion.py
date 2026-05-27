#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _traveling_companion.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/8/7 21:37
import math
from collections import defaultdict
from typing import Dict, Set, List

from moveminerx.mining._basic_class import MovingObject, Snapshot, SnapshotBuilder, \
    TravelingCompanion, TravelingCompanionCandidate, TravelingBuddy, ClusterClass


class TravelingCompanionMiner:
    """Base class for traveling companion mining algorithms"""

    def __init__(self, objects: Dict[str, MovingObject], min_size: int, min_duration: int,
                 eps: float, minPts: int):
        """
        :param objects:
        """
        self.objects = objects
        self.eps = eps  # ε in paper (distance threshold)
        self.minPts = minPts  # μ in paper (density threshold)
        self.min_duration = min_duration  # δt in paper
        self.min_size = min_size  # δs in paper

        # Data structures to maintain state
        self.snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(objects)  # timestamp->snapshot
        self.candidates: Dict[int, TravelingCompanionCandidate] = {}  # candidate_id -> candidate
        self.next_candidate_id = 1

    def process_snapshot(self, snapshot: Snapshot):
        """Process a new snapshot (to be implemented by subclass)"""
        raise NotImplementedError

    def _create_candidate(self, objects: Set[str], timestamp: int) -> int:
        """Create a new companion candidate"""
        candidate_id = self.next_candidate_id
        self.next_candidate_id += 1
        candidate = TravelingCompanionCandidate(objects=objects.copy(), duration=1, first_seen=timestamp, timestamps=[timestamp])
        self.candidates[candidate_id] = candidate
        return candidate_id

    def _update_candidate(self, candidate_id: int, new_objects: Set[str], timestamp: int):
        """Update an existing candidate with new object set"""
        if candidate_id not in self.candidates:
            return

        candidate = self.candidates[candidate_id]
        candidate.objects = new_objects
        candidate.duration = timestamp - candidate.first_seen + 1
        candidate.timestamps.append(timestamp)

    def _remove_candidate(self, candidate_id: int):
        """Remove a candidate from tracking"""
        if candidate_id in self.candidates:
            del self.candidates[candidate_id]

    def _find_companions(self) -> List[TravelingCompanionCandidate]:
        """Find all candidates that qualify as traveling companions"""
        return [cand for cand in self.candidates.values() if cand.is_companion(self.min_size, self.min_duration)]


class ClusteringAndIntersectionMiner(TravelingCompanionMiner):
    """
    Basic clustering and intersection method for traveling companion detection
    It is just similar to the CMC algorithm for convoy detection
    """

    # def __init__(self, objects: Dict[str, MovingObject],  min_size: int, min_duration: int,
    #              eps: float, minPts: int):
    #     super().__init__(objects=objects, min_size=min_size, min_duration=min_duration, eps=eps, minPts=minPts)

    def process_snapshot(self, snapshot: Snapshot) -> List[TravelingCompanionCandidate]:
        """Process a new snapshot using Clustering and intersection method"""
        # Cluster the objects in this snapshot
        clusters = snapshot.cluster_snapshot(snapshot, eps=self.eps, minPts=self.minPts)
        self.snapshots[snapshot.time] = snapshot

        new_candidates: Dict[int, TravelingCompanionCandidate] = {}
        # new_candidates = set()
        Output_TCompanions: Dict[int, TravelingCompanionCandidate] = {}  #
        # print(len(self.candidates))
        # Intersect existing candidates with new clusters
        for candidate_id, candidate in list(self.candidates.items()):
            self.candidates[candidate_id].is_closed = False
            max_intersection = set()
            for cid, cluster in clusters.items():
                # print(cluster)
                intersection = set(cluster.members) & candidate.objects
                # if len(intersection) >= self.min_size:
                #     new_candidates.add(candidate)
                if len(intersection) > len(max_intersection):
                    max_intersection = intersection
                    cluster.assigned = True

            if len(max_intersection) >= self.min_size:
                # Update candidate with intersection results
                self._update_candidate(candidate_id, max_intersection, snapshot.time)
                new_candidates[candidate_id] = self.candidates[candidate_id]
            else:
                # Candidate no longer meets size threshold
                self.candidates[candidate_id].is_closed = True
                # self._remove_candidate(candidate_id=candidate_id)
                # if candidate.duration >= self.min_duration:
                #     Output_TCompanions[candidate_id] = candidate
                # else:
                #     self._remove_candidate(candidate_id=candidate_id)

        for cid, cluster in clusters.items():
            if not cluster.assigned:
                if len(cluster.members) >= self.min_size:
                    candidate_id = self._create_candidate(set(cluster.members), snapshot.time)
                    new_candidates[candidate_id] = self.candidates[candidate_id]

        found_candidates = self._find_companions()
        self.candidates = new_candidates
        # 批量输出而非即时
        # 确保候选在被移除前已通过其他途径检查过duration (上一时刻)
        return found_candidates


class SmartAndClosedMiner(TravelingCompanionMiner):
    """Improved smart-and-closed algorithm with early termination and closed candidates"""

    def is_closed(self, candidate: TravelingCompanionCandidate, existing: Dict[int, TravelingCompanionCandidate]):
        for cand_id, other in existing.items():
            if candidate.objects.issubset(other.objects) and candidate.duration <= other.duration:
                return False
            else:
                return True

    def process_snapshot(self, snapshot: Snapshot) -> List[TravelingCompanionCandidate]:
        """Process a new snapshot using smart-and-closed method"""
        # CLuster the objects in this snapshot
        clusters = snapshot.cluster_snapshot(snapshot, eps=self.eps, minPts=self.minPts)
        self.snapshots[snapshot.time] = snapshot

        new_candidates: Dict[int, TravelingCompanionCandidate] = {}
        Qualified_TCompanions: Dict[int, TravelingCompanionCandidate] = {}  #
        # Process existing candidates with smart intersection
        for candidate_id, candidate in self.candidates.items():
            remaining_objects = candidate.objects.copy()
            extended = False
            # traverse the cluster of the moving object points in current timestamp
            for cid, cluster in clusters.items():
                # Early termination if remaining objects can't meet size threshold
                # if there are more than size(tc_persist_members) − m objects of tc_persist_members already appearing in intersected clusters,
                # continuously intersecting r with remaining clusters will not generate any meaningful results with size larger than m.
                if len(remaining_objects) < self.min_size:
                    break
                # calculate the intersection of current cluster c and existing traveling companion tc
                intersection = set(cluster.members) & set(remaining_objects)
                # remove intersected objects from remaining objects
                remaining_objects -= intersection

                if len(intersection) >= self.min_size:
                    self._update_candidate(candidate_id, intersection, snapshot.time)
                    new_candidates[candidate_id] = self.candidates[candidate_id]
                    # if the members of the cluster have appeared in existing traveling companions, we think that is closed
                    # and will not be added into the C_next_traveling_companions and vice verse.
                    if cluster.size() == len(intersection):
                        clusters[cid].closed = False
                    # 输出当前时刻满足要求的companion, 可以不用, 这个是streaming mining, 当前时刻更新的结果, 全部会去进行判断是否输出
                    # if self.candidates[candidate_id].duration >= self.min_duration:
                    #     Qualified_TCompanions[candidate_id] = self.candidates[candidate_id]
                    extended = True

            if not extended:
                self.candidates[candidate_id].is_closed = True
                # self._remove_candidate(candidate_id)
                # continue
                #
                # if candidate.duration >= self.min_duration:
                #     Qualified_TCompanions[candidate_id] = candidate
                # else:
                #     # Candidate no longer meets size threshold
                #     self._remove_candidate(candidate_id)
                #     continue

            # Check if this is a closed candidate
            if self.is_closed(candidate, self.candidates):
                self.candidates[candidate_id].is_closed = True

        # Add new candidates from this snapshot's clusters (only closed ones)
        for cid, cluster in clusters.items():
            if len(cluster.members) >= self.min_size:
                # Check if this cluster is already covered by an existing closed candidate
                # 不会添加一个对象集合完全相同但持续时间更短的候选 因为已有更优的closed候选存在
                is_closed = True
                for cand_id, candidate in self.candidates.items():
                    if set(cluster.members).issubset(self.candidates[cand_id].objects):
                        is_closed = False
                        break
                if is_closed:
                    candidate_id = self._create_candidate(set(cluster.members), snapshot.time)
                    new_candidates[candidate_id] = self.candidates[candidate_id]

        # self.candidates = new_candidates
        found_candidates = self._find_companions()
        self.candidates = new_candidates
        # 批量输出而非即时
        # 确保候选在被移除前已通过其他途径检查过duration (上一时刻)
        return found_candidates

        # return self._find_companions()


class TravelBuddyBasedMiner(TravelingCompanionMiner):
    """Traveling buddy-based companion discovery algorithm"""

    def __init__(self, objects: Dict[str, MovingObject], min_size: int, min_duration: int,
                 eps: float, minPts: int, buddy_radius: float):
        super().__init__(objects, min_size, min_duration, eps, minPts)
        self.buddy_radius = buddy_radius  # δγ in paper
        self.buddies: Dict[int, TravelingBuddy] = {}  # buddy_id -> buddy
        self.next_buddy_id = 1
        self.object_to_buddy: Dict[str, int] = {}  # object_id -> buddy_id

    def process_snapshot(self, snapshot: Snapshot) -> List[TravelingCompanionCandidate]:
        """Process a new snapshot using buddy-based method for traveling companion discovery"""
        # Update or initialized buddies
        self._update_buddies(snapshot)

        # Perform buddy-based clustering
        clusters = self._buddy_based_clustering(snapshot)

        snapshot.clusters = [ClusterClass(cid=str(i), members=list(cluster), t=snapshot.time) for i, cluster in
                             enumerate(clusters)]
        self.snapshots[snapshot.time] = snapshot
        new_candidates = {}
        closed_candidates = set()  # Track closed candidates to avoid duplicates

        # Process existing candidates with buddy-based intersection
        for candidate_id, candidate in list(self.candidates.items()):
            self.candidates[candidate_id].is_closed = False
            if len(candidate.objects) < self.min_size:
                continue
            extended = False
            remaining_objects = candidate.objects.copy()
            # Check which objects in candidate are in current clusters
            for cluster in snapshot.clusters:

                if len(remaining_objects) < self.min_size:
                    break

                intersection = candidate.objects & set(cluster.members)
                remaining_objects -= intersection

                if len(intersection) >= self.min_size:
                    # Update candid_update_candidate with remaining objects
                    self._update_candidate(candidate_id, intersection, snapshot.time)
                    new_candidates[candidate_id] = self.candidates[candidate_id]
                    extended = True

                # Check if this is a closed candidate
                # is_closed = True
                # for other_id, other_cand in self.candidates.items():
                #     if other_id != candidate_id and remaining_objects.issubset(other_cand.objects) \
                #             and self.candidates[candidate_id].duration <= other_cand.duration:
                #         is_closed = False
                #         break
                # if is_closed:
                #     closed_candidates.add(candidate_id)
            if not extended:
                # Candidate no longer meets size threshold:
                # self._remove_candidate(candidate_id)
                self.candidates[candidate_id].is_closed = True

        # Add new candidates from this snapshot's clusters (only closed ones)
        for cluster in snapshot.clusters:
            if len(cluster.members) >= self.min_size:
                # Check if this cluster is already covered by an existing closed candidate
                is_closed = True
                for cand_id, candidate in self.candidates.items():
                    if set(cluster.members).issubset(candidate.objects):
                        is_closed = False
                        break
                if is_closed:
                    candidate_id = self._create_candidate(set(cluster.members), snapshot.time)
                    new_candidates[candidate_id] = self.candidates[candidate_id]
                    # closed_candidates.add(candidate_id)

        # self.candidates = new_candidates
        found_candidates = self._find_companions()
        self.candidates = new_candidates
        # 批量输出而非即时
        # 确保候选在被移除前已通过其他途径检查过duration (上一时刻)
        return found_candidates

    def _update_buddies(self, snapshot: Snapshot):
        """
        Maintain traveling buddies for the new snapshot. It is corresponding to the algorithm 3 in the paper.
        Parameters:
            snapshot: the next/coming snapshot moving object points
        Returns:
            Updated traveling companion set of current snapshot s.
        """
        # First pass: Update buddy centers based on object movements
        new_positions = snapshot.points
        # Track which objects have moved too far from their buddies
        to_split = defaultdict(list)
        for buddy_id, buddy in self.buddies.items():
            old_center = buddy.center
            # update the buddy center and buddy radius using new positions
            # must first update the center of the buddy and then update the lambda_ distance of the buddy.
            # It is because the calculation of lambda distance requires the center coordinate.
            buddy.update_center(new_positions)
            # Check if any objects need to be split from this buddy
            for obj_id in list(buddy.objects):
                if obj_id in new_positions.keys():
                    dist = math.sqrt((new_positions[obj_id].x - buddy.center.x) ** 2 + (
                            new_positions[obj_id].y - buddy.center.y) ** 2)
                    if dist > self.buddy_radius:
                        # split obj_id out as a new buddy bj
                        to_split[buddy_id].append(obj_id)

        # Perform splits
        for buddy_id, objects_to_split in to_split.items():
            if buddy_id not in self.buddies:
                continue

            buddy = self.buddies[buddy_id]
            for obj_id in objects_to_split:
                # remove the object in the original buddy
                buddy.objects.remove(obj_id)
                del self.object_to_buddy[obj_id]

                # Create a new buddy for this object
                new_buddy_id = self.next_buddy_id
                self.next_buddy_id += 1
                new_buddy = TravelingBuddy(buddy_id=new_buddy_id, objects={obj_id}, center=new_positions[obj_id],
                                           radius=0.0)

                self.buddies[new_buddy_id] = new_buddy
                self.object_to_buddy[obj_id] = new_buddy_id

            # Update the original buddy's center after splits
            if buddy.objects:
                buddy.update_center(new_positions)
            else:
                # Remove empty buddy
                del self.buddies[buddy_id]

        # Second pass: Merge close buddies (Merge operation)
        buddy_ids = list(self.buddies.keys())
        merged = set()

        for i in range(len(buddy_ids)):
            buddy1_id = buddy_ids[i]
            if buddy1_id in merged:
                continue

            buddy1 = self.buddies[buddy1_id]
            for j in range(i + 1, len(buddy_ids)):
                buddy2_id = buddy_ids[j]
                if buddy2_id in merged:
                    continue
                buddy2 = self.buddies[buddy2_id]

                # Check merge condition
                center_dist = buddy1.center.distance_to(buddy2.center)
                if center_dist + buddy1.radius + buddy2.radius <= 2 * self.buddy_radius:
                    # Merge buddy2 to buddy1
                    buddy1.objects.update(buddy2.objects)
                    for obj_id in buddy2.objects:
                        self.object_to_buddy[obj_id] = buddy1_id

                    # Update center as weighted average
                    buddy1.update_center(new_positions)

                    # Merge candidate IDs
                    buddy1.candidate_ids.update(buddy2.candidate_ids)

                    # Mark buddy2 for removal
                    merged.add(buddy2_id)

        # Remove merged buddies
        for buddy_id in merged:
            del self.buddies[buddy_id]

        # Initialize buddies for new objects not in any buddy
        for obj_id in snapshot.points.keys():
            if obj_id not in self.object_to_buddy:
                new_buddy_id = self.next_buddy_id
                self.next_buddy_id += 1
                new_buddy = TravelingBuddy(buddy_id=new_buddy_id, objects={obj_id}, center=new_positions[obj_id],
                                           radius=0.0)
                self.buddies[new_buddy_id] = new_buddy
                self.object_to_buddy[obj_id] = new_buddy_id

    def _buddy_based_clustering(self, snapshot: Snapshot) -> List[Set[str]]:
        """Perform buddy-based clustering on the snapshot"""
        clusters = []
        visited_buddies = set()
        visited_objects = set()

        # First, identify density-connected buddies (Lemma2)
        density_connected_buddies = [
            buddy for buddy in self.buddies.values()
            if (len(buddy.objects) >= self.minPts + 1 and
                buddy.radius <= self.eps / 2)
        ]

        # For each density-connected buddy, all its objects form a cluster
        for buddy in density_connected_buddies:
            clusters.append(buddy.objects.copy())
            visited_buddies.add(buddy.buddy_id)
            visited_objects.update(buddy.objects)

        # Now process remaining objects and buddies
        buddy_list = [buddy for buddy in self.buddies.values() if buddy.buddy_id not in visited_buddies]

        while buddy_list:
            current_buddy = buddy_list.pop()
            if current_buddy.buddy_id in visited_buddies:
                continue

            # Start a new cluster with this buddy's objects
            cluster = set(current_buddy.objects)
            visited_buddies.add(current_buddy.buddy_id)
            visited_objects.update(current_buddy.objects)

            # Find density-connected buddies (Lemma 4)
            neighbors = []
            for other_buddy in buddy_list:
                if other_buddy.buddy_id in visited_buddies:
                    continue

                # Check if buddies are two far apart (Lemma 3)
                center_dist = current_buddy.center.distance_to(other_buddy.center)
                if center_dist - current_buddy.radius - other_buddy.radius > self.eps:
                    continue  # Too far apart, skip

                # Check if any pair of objects from the two buddies are enougth
                found_connection = False

                for obj1 in current_buddy.objects:
                    if obj1 not in snapshot.points.keys():
                        continue

                    obj1_pt = snapshot.points[obj1]

                    for obj2 in other_buddy.objects:
                        if obj2 not in snapshot.points.keys():
                            continue
                        obj2_pt = snapshot.points[obj2]

                        if obj1_pt.distance_to(obj2_pt) <= self.eps:
                            found_connection = True
                            break

                    if found_connection:
                        break

                if found_connection:
                    neighbors.append(other_buddy)
                    cluster.update(other_buddy.objects)
                    visited_buddies.add(other_buddy.buddy_id)
                    visited_objects.update(other_buddy.objects)

            if len(cluster) >= self.minPts:
                clusters.append(cluster)

        # Now process any remaining objects not in any buddy (shouldn't happen with proper maintenance
        remaining_objects = set(snapshot.points.keys()) - visited_objects

        if remaining_objects:
            # Perform regular DBSCAN on remaining objects
            temp_snapshot = Snapshot(time=snapshot.time)
            temp_snapshot.points = {k: snapshot.points[k] for k in remaining_objects}
            temp_clusters = temp_snapshot.cluster_snapshot(temp_snapshot, eps=self.eps, minPts=self.minPts)

            for cluster in temp_clusters.values():
                if len(cluster.members) >= self.min_size:
                    clusters.append(cluster.closed)

        return clusters


def run_online_traveling_companion_discovery(objects: Dict[str, MovingObject], min_size: int, min_duration: int,
                                             eps: float, minPts: int, buddy_radius: float = 0, method='regular'):
    Results = []
    if method == 'regular':
        miner = ClusteringAndIntersectionMiner(objects=objects, min_size=min_size, min_duration=min_duration, eps=eps,
                                               minPts=minPts)
    elif method == 'smart':
        miner = SmartAndClosedMiner(objects=objects, min_size=min_size, min_duration=min_duration, eps=eps,
                                               minPts=minPts)

    elif method == 'buddy':
        miner = TravelBuddyBasedMiner(objects=objects, min_size=min_size, min_duration=min_duration, eps=eps,
                                      minPts=minPts, buddy_radius=buddy_radius)
    else:
        raise ValueError('please select correct method')

    for t, snapshot in miner.snapshots.items():
        companions = miner.process_snapshot(snapshot=snapshot)
        Results.extend([companion for companion in companions if companion.is_closed])
        # print(t, companions)
        # print('*'*20)
    Results.extend([companion for companion in companions])
    return Results
