#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _evolving_convoy.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/8/14 13:33
from typing import List, Dict, Set
# from previous.basic_Class import MovingObject
from moveminerx.mining._basic_class import EvolvingConvoy, Snapshot, SnapshotBuilder, MovingObject, DynamicConvoy, ClusterClass


class EvolvingConvoysMiner:
    """Base class for evolving convoys discovery algorithms"""

    def __init__(self, objects: Dict[str, MovingObject], m: int = 2, k: int = 2, w: int = 5, eps: float = 20,
                 minPts: int = 2):
        """
        Initialize the discoverer with parameters:
        - objects:
        - eps: DBSCAN epsilon parameter (spatial proximity)
        - min_pts: DBSCAN min_points parameter
        - m: minimum number of persistent members
        - k: minimum number of times a dynamic member must be connected in a w-window
        - w: minimum duration for a convoy
        """
        self.objects = objects
        self.m = m
        self.k = k
        self.w = w
        self.eps = eps
        self.minPts = minPts
        self.snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(self.objects)

    def run(self) -> List[EvolvingConvoy]:
        """Discover evolving convoys from moving objects data"""
        raise NotImplementedError


class SimpleSliceBySlice(EvolvingConvoysMiner):
    """Implementation of the S^3 algorithm from the paper"""

    def run(self) -> List[EvolvingConvoy]:
        # Create a dictionary for quick object access
        all_object_ids = {oid for oid, obj in self.objects.items()}
        evolving_convoys = []
        convoys_active: List[EvolvingConvoy] = []  # List of DynamicConvoy objects being tracked
        # snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(objects)
        snapshots = self.snapshots
        sorted_times = sorted(snapshots.keys())

        for t in sorted_times:
            snapshot = snapshots[t]
            clusters = snapshot.cluster_snapshot(snapshot, eps=self.eps, minPts=self.minPts)
            Convoys_next = []
            for i, e_convoy in enumerate(convoys_active):
                e_convoy.stages[-1].extended = False
                for cid, cluster in clusters.items():
                    # Check if at least m persistent members are in this cluster
                    convoy = e_convoy.stages[-1]

                    if len(convoy.persistent_members & set(cluster.members)) >= self.m:
                        if convoy.duration() >= self.w:
                            new_convoy = DynamicConvoy(persistent_members=set(cluster.members), start_time=t,
                                                       end_time=t, dynamic_members=set(),
                                                       time_to_objects={t: cluster.members})
                            e_convoy.stages.append(new_convoy)
                        else:
                            # Extend the convoy with this cluster
                            convoy = self._extend_convoy(convoy, cluster, t, all_object_ids)
                            e_convoy.stages[-1] = convoy
                        e_convoy.end_time = t
                        Convoys_next.append(e_convoy)
                        e_convoy.stages[-1].extended = True
                        cluster.assigned = True
                        break

                if not e_convoy.stages[-1].extended:
                    evolving_convoys.append(e_convoy)

            convoys_active = Convoys_next

            # Find new convoys starting in previous N partitions
            for cid, cluster in clusters.items():
                if not cluster.assigned and len(cluster.members) >= self.m:
                    new_convoy = DynamicConvoy(persistent_members=set(cluster.members), start_time=t,
                                               end_time=t, dynamic_members=set(),time_to_objects={t: cluster.members})
                    new_evolving_convoy = EvolvingConvoy(stages=[new_convoy], start_time=t, end_time=t)
                    convoys_active.append(new_evolving_convoy)
            # print('t end', len(convoys_active), convoys_active, )

        # After processing all timestamps, add remaining convoys that meet duration
        for e_convoy in convoys_active:
            if e_convoy.stages[-1].duration() >= self.w:
                evolving_convoys.append(e_convoy)
            elif len(e_convoy.stages) > 1:
                e_convoy.stages.pop()
                evolving_convoys.append(e_convoy)
        return evolving_convoys

    def _extend_convoy(self, convoy: DynamicConvoy, cluster: ClusterClass, current_time: int, object_ids: Set[str]):
        """Extend a convoy with a new cluster observation"""
        convoy.end_time = current_time
        convoy.time_to_objects[current_time] = cluster.members
        # update persistent and dynamic members based on their participation
        # For each object in the system, count its participation in the last w timestamps
        start_t = max(convoy.start_time, current_time - self.w + 1)
        for o in cluster.members:
            count = 0
            if o in convoy.persistent_members:
                convoy.persistent_members.add(o)
            else:
                for t in range(start_t, current_time):
                    if o in convoy.time_to_objects[t]:
                        count += 1
            if count >= self.k:
                convoy.dynamic_members.add(o)
            else:
                if o in convoy.dynamic_members:
                    convoy.dynamic_members.remove(o)
        return convoy
