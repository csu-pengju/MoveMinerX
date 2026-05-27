#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _loose_group_companion.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/8/13 12:34
import copy
from typing import Dict, List, Tuple, Set

from moveminerx.mining._basic_class import MovingObject, SnapshotBuilder, Snapshot, MicroGroup, ClusterClass, LooseGroupCandidate, LooseGroupCompanion


class LooseGroupCompanionMiner:
    """Main class implementing the loose group companion discovery algorithm

    Parameters
    ----------
    objects: Dict[str, MovingObject]
        moving objects data {o1: MovingObject, o2: MovingObject, ....}
    delta_s: int
        Candidate size threshold
    delta_t: int
        Candidate duration threshold
    delta_l: int
     Candidate leave time threshold
    eps: float
        Micro-group distance threshold
    gamma:int
      Micro-group size threshold
    """

    def __init__(self, objects: Dict[str, MovingObject], delta_s: int = 2, delta_t: int = 2, delta_l: int = 5,
                 eps: float = 20, gamma: int = 2):
        self.objects = objects
        self.delta_s = delta_s
        self.delta_t = delta_t
        self.delta_l = delta_l
        self.eps = eps
        # self.minPts = minPts
        self.gamma = gamma

    def run(self):

        objects, delta_s, delta_t, delta_l, eps, gamma = self._check_params(self.objects, self.delta_s, self.delta_t,
                                                                            self.delta_l, self.eps, self.gamma)

        snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(objects)
        sorted_times = sorted(snapshots.keys())
        LGCP_closed: List[LooseGroupCompanion] = []
        LGCP_active: List[LooseGroupCandidate] = []  # current active loose group companions

        for i, t in enumerate(sorted_times):
            snapshot = snapshots[t]
            if i == 0:
                # 初始化micro-groups
                G, U = self._create_micro_groups(snapshot, eps, gamma)
            else:
                G = self._maintain_micro_groups(G, snapshot, eps, gamma)
                U = set(snapshot.points.keys()) - set().union(*[g.member_ids for g in G])

            clusters = self._micro_group_based_clustering(G, U, snapshot, eps, gamma)
            # Step 2: Candidate extension
            new_candidates = []
            for candidate in LGCP_active:
                if candidate.size() < delta_s:
                    continue  # Lemma 1
                remaining_members = set(candidate.member_ids)

                for cluster in clusters:
                    if cluster.size() < delta_s:
                        continue  # Lemma2

                    # Intersect candidate with cluster
                    cluster_members = cluster.members
                    intersection = set(remaining_members) & set(cluster_members)
                    if len(intersection) < delta_s:
                        continue  # Lemma3
                    # Update leave times for members not in intersection
                    updated_members = {}
                    for obj_id in candidate.member_ids:
                        if obj_id in intersection:
                            updated_members[obj_id] = candidate.leave_times.get(obj_id, 0)  # Leave time maintains
                        else:
                            # Increment leave time
                            updated_members[obj_id] = candidate.leave_times.get(obj_id, 0) + 1

                    # Prune members who have left for too long
                    pruned_members = {
                        obj_id: leave_time for obj_id, leave_time in updated_members.items() if
                        leave_time <= self.delta_l
                    }
                    # Create new candidate with updated info
                    new_duration = candidate.duration + 1
                    new_candidate = LooseGroupCandidate(member_ids=set(pruned_members.keys()), duration=new_duration,
                                                        leave_times=pruned_members, start_time=candidate.start_time,
                                                        end_time=t)
                    new_candidates.append(new_candidate)
                    # if new_duration >= self.delta_t:
                    #     # Convert to companion
                    #     LGCP_closed.append(LooseGroupCompanion(
                    #         member_ids=new_candidate.member_ids,
                    #         duration=new_candidate.duration
                    #     ))
                    # else:
                    #     new_candidates.append(new_candidate)
                    cluster.assigned = True
                    # Remove intersected members from consideration
                    remaining_members -= intersection
                    candidate.extended = True

                if not candidate.extended and candidate.duration >= self.delta_t:
                    # Convert to companion
                    LGCP_closed.append(LooseGroupCompanion(
                        member_ids=candidate.member_ids,
                        duration=candidate.duration,
                        timestamps=[t for t in range(candidate.start_time, candidate.end_time + 1)]
                    ))
            # Step 3: Candidate creation
            for cluster in clusters:
                if not cluster.assigned and cluster.size() >= self.delta_s and self._is_closed_cluster(cluster):
                    # Create a new candidate from closed cluster
                    new_candidate = LooseGroupCandidate(
                        member_ids=set(cluster.members),
                        duration=1,
                        leave_times={obj_id: 0 for obj_id in cluster.members}, start_time=t, end_time=t)

                    new_candidates.append(new_candidate)

            # Update state for next snapshot
            LGCP_active = new_candidates

        # 收尾
        for candidate in LGCP_active:
            if candidate.duration >= self.delta_t and candidate.size() >= self.delta_s:
                # LGCP_closed.append(candidate)
                LGCP_closed.append(LooseGroupCompanion(
                    member_ids=candidate.member_ids,
                    duration=candidate.duration,
                    timestamps=[t for t in range(candidate.start_time, candidate.end_time + 1)]
                ))

        # Return any newly discovered companions
        return [c for c in LGCP_closed if c.duration >= self.delta_t]

    def _create_micro_groups(self, snapshot: Snapshot, eps: float, gamma: int) -> Tuple[List[MicroGroup], Set[str]]:
        """
        Initialize micro-groups for the first snapshot (Algorithm 2)
        Parameters:
            - snapshot: current snapshot objects
            - eps: distance threshold
            - gamma: minimum micro-group size
        Returns:
            G: List of micro groups,
            U: remaining objects that are not in G
        """
        visited = set()
        micro_groups = []
        U = set()
        for oid, pt in snapshot.points.items():
            if oid in visited:
                continue
            visited.add(oid)

            # seeking neighbors
            neighbors = [nid for nid, npt in snapshot.points.items()
                         if pt.distance_to(npt) <= eps]
            if len(neighbors) >= gamma:  # -1 because rep is not in neighbors
                # Create micro group
                radius = max(pt.distance_to(snapshot.points[nid]) for nid in neighbors)
                mg = MicroGroup(rep_id=oid, member_ids=set(neighbors), radius=radius)
                micro_groups.append(mg)
                visited.update(neighbors)
            else:
                U.add(oid)

        return micro_groups, U

    def _maintain_micro_groups(self, prev_micro_groups: List[MicroGroup], snapshot: Snapshot, eps: float, gamma: int) -> \
            List[MicroGroup]:
        """
        Maintain micro-groups for subsequent snapshots (Algorithm 3)
        """
        micro_groups = []
        objects_in_snapshot = set(snapshot.points.keys())
        processed_objects = set()
        remaining_objects = set(oid for oid in snapshot.points.keys())
        for old_mg in prev_micro_groups:
            if old_mg.rep_id not in objects_in_snapshot:
                # Representative disappeared - micro-group splits
                # Need to reinitialize these objects
                remaining_objects.update(old_mg.get_all_members())
                continue

            rep_pt = snapshot.points[old_mg.rep_id]
            # looking for new neighbors
            neighbors = [oid for oid, pt in snapshot.points.items()
                         if rep_pt.distance_to(pt) <= eps]

            # Check if micro-group survives
            if len(neighbors) >= gamma:
                # Check for merge with other micro-groups
                merged = False
                for other_rep_id in neighbors:
                    # Check if this neighbor is a rep of another micro-group
                    for mg in micro_groups:
                        if mg.rep_id == other_rep_id:
                            # Merge micro-groups
                            new_members = copy.copy(mg.member_ids)
                            new_members.union(neighbors)
                            # new_members = mg.member_ids.union(neighbors)
                            new_members.discard(old_mg.rep_id)
                            new_members.discard(other_rep_id)
                            #  Calculate new radius
                            if len(new_members) < 1:
                                continue
                            radius = max([rep_pt.distance_to(snapshot.points[nid]) for nid in new_members])
                            # Create merged micro-group
                            merged_mg = MicroGroup(rep_id=old_mg.rep_id, member_ids=new_members, radius=radius)
                            micro_groups.append(merged_mg)
                            micro_groups.remove(mg)
                            merged = True
                            break
                    if merged:
                        break
                if not merged:
                    # Micro-group survives without merging
                    radius = max(rep_pt.distance_to(snapshot.points[nid]) for nid in neighbors)
                    micro_group = MicroGroup(rep_id=old_mg.rep_id, member_ids=set(neighbors), radius=radius)
                    micro_groups.append(micro_group)
                processed_objects.add(old_mg.rep_id)
                processed_objects.union(neighbors)
                remaining_objects.difference_update(neighbors)
                remaining_objects.discard(old_mg.rep_id)

        # Handle remaining objects that weren't part of any maintained micro-group
        if remaining_objects:
            # Need to initialize new micro-groups with remaining objects
            # This is essentially the same as Algorithm 2 but with remaining_objects
            temp_snapshot = Snapshot(
                time=snapshot.time)
            # objects = [oid for oid in snapshot.points.keys() if oid in remaining_objects]
            temp_snapshot.points = {
                oid: pt for oid, pt in snapshot.points.items() if oid in remaining_objects
            }
            new_micro_groups, _ = self._create_micro_groups(temp_snapshot, eps, gamma)

            micro_groups.extend(new_micro_groups)
        return micro_groups

    def _micro_group_based_clustering(self, micro_groups: List[MicroGroup], U, snapshot: Snapshot, eps: float,
                                      gamma: int) \
            -> List[ClusterClass]:
        """
          Perform micro-group based clustering (Algorithm 4)
        """
        clusters = []
        visited = set()
        # objects_in_snapshot = set(snapshot.points.keys())

        # First cluster micro-groups
        for mg_i in micro_groups:
            if mg_i.rep_id in visited:
                continue

            c_members = mg_i.member_ids
            visited.update(mg_i.member_ids)
            for mg_k in micro_groups:
                if mg_k.rep_id in visited:
                    continue

                if snapshot.points[mg_k.rep_id].distance_to(snapshot.points[mg_i.rep_id]) <= mg_i.radius + mg_k.radius:
                    c_members |= mg_k.member_ids
                    visited.update(mg_k.member_ids)

            # 加入U中满足条件的对象
            for u in list(U):
                for oid in c_members:
                    if snapshot.points[oid].distance_to(snapshot.points[u]) <= eps:
                        c_members.add(u)
                        U.remove(u)
                        break

            clusters.append(ClusterClass(cid='-1', t=snapshot.time, members=list(c_members)))

        return clusters

    def _is_closed_cluster(self, cluster: ClusterClass) -> bool:
        """Check if a cluster is closed (Definition 7)"""
        return cluster.size() >= self.delta_s

    def _check_params(self, objects: Dict[str, MovingObject], delta_s: int, delta_t: int, delta_l: int, eps: float,
                      gamma: int):
        if objects is None:
            objects = self.objects
        if delta_s is None:
            delta_s = self.delta_s
        if delta_l is None:
            delta_l = self.delta_l
        if delta_t is None:
            delta_t = self.delta_t
        if eps is None:
            eps = self.eps
        if gamma is None:
            gamma = self.gamma

        return objects, delta_s, delta_t, delta_l, eps, gamma
