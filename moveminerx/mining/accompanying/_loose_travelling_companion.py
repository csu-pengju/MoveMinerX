#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _loose_travelling_companion.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/8/13 12:34
import random
from itertools import combinations
from typing import Dict, List, Set

from moveminerx.mining._basic_class import MovingObject, Snapshot, SnapshotBuilder, LTCPPattern, ClusterSet, ClusterClass


class LooseTravelingCompanionMiner:
    """Discover closed Loose Traveling Companion Patterns (LTCP) per Naserian et al. (2016).

    Parameters
    ----------
    objects: Dict[str, MovingObject]
        moving objects data {o1: MovingObject, o2: MovingObject, ....}
    mG : int
        Minimum group size (|OG| ≥ mG)
    dG : int
        Minimum duration in *number of slots included in the pattern* (≥ dG)
    fC : int
        Minimum number of slots with full gathering (all OG in one cluster)
    eps: float
        The neighboring radius required by DBSCAN
    minPts: int
        Minimum number of objects required by DBSCAN
    max_group_size : int
        Upper bound on group size when enumerating cluster-set candidates (speeds up subset generation).
    """

    def __init__(self, objects: Dict[str, MovingObject], mG: int = 2, dG: int = 2, fC: int = 5, eps: float = 20,
                 min_pts: int = 2,
                 max_group_size: int = 100):
        self.objects = objects
        self.mG = mG
        self.dG = dG
        self.fC = fC
        self.eps = eps
        self.min_pts = min_pts
        self.max_group_size = max_group_size

    def run(self):
        snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(self.objects)
        sorted_times = sorted(snapshots.keys())
        LTCP_closed: List[LTCPPattern] = []
        LTCP_active: List[LTCPPattern] = []  # current active loose traveling companions

        for t in sorted_times:
            snapshot = snapshots[t]
            clusters = snapshot.cluster_snapshot(snapshot, eps=self.eps, minPts=self.min_pts)  # clusters at t
            LTCP_next: List[LTCPPattern] = []
            # 1) Extend existing candidates
            for ltcp in LTCP_active:
                ltcp.extended = False
                temp_cluster_sets = []
                for cid, cluster in clusters.items():
                    # Find all clusters that are subsets of the candidate's object group
                    if set(cluster.members).issubset(ltcp.OG):
                        temp_cluster_sets.append(cluster)
                # if t>7:
                #     print('t', t, ltcp.OG, temp_cluster_sets)
                # Check if the union of these clusters equals the candidate's object group
                if temp_cluster_sets:
                    object_union = set()
                    for temp in temp_cluster_sets:
                        object_union.update(set(temp.members))
                    # if t >7
                    #     print('object_union', object_union, object_union==ltcp.OG)
                    if object_union == ltcp.OG and len(temp_cluster_sets) > 0:
                        cs = ClusterSet.from_clusters(t, temp_cluster_sets)
                        ltcp_next = LTCPPattern(object_group=ltcp.OG, cluster_sequences=ltcp.TS + [cs])
                        ltcp_next.valid = self._is_valid(ltcp_next)
                        ltcp.extended = True
                        LTCP_next.append(ltcp_next)

                if not ltcp.extended:
                    # Not extended: if valid and cannot extend further, consider as closed now
                    if ltcp.valid and not self._exists_sup_extension(ltcp, list(clusters.values()), t):
                        LTCP_closed.append(ltcp)

            # Create new candidates from closed subset-collection at t
            new_candidates = self._generate_closed_cluster_sets(list(clusters.values()), LTCP_next)
            # if t > 7:
            #     print('new_candidates', new_candidates)
            for cs in new_candidates:
                if len(cs.union_members) >= self.mG:
                    new_ltcp = LTCPPattern(object_group=cs.union_members, cluster_sequences=[cs], valid=False)
                    LTCP_next.append(new_ltcp)

            LTCP_active = list(set(LTCP_next))

        # Finalize: any remaining valid, non-extendable candidates are closed
        for ltcp in LTCP_active:
            if ltcp.valid:
                LTCP_closed.append(ltcp)
        return self._deduplicate(LTCP_closed)

    def _is_valid(self, ltcp_next: LTCPPattern) -> bool:

        if len(ltcp_next.OG) >= self.mG and len(ltcp_next.TS) >= self.dG and ltcp_next.freq_full_gathering() >= self.fC:
            return True
        return False

    def _exists_sup_extension(self, ltcp: LTCPPattern, clusters: List[ClusterClass], t: int):
        """
        Heuristic:
        returns True if the loose traveling companion pattern (ltcp) is closed, i.e., there is not any way to extend P at time t.
        otherwise False (ltcp is not closed yet at this t), i.e., an extension exist.
        """
        temp = [c for c in clusters if set(c.members).issubset(ltcp.OG)]
        union: Set[str] = set()
        for c in temp:
            union.update(c.members)
        return union == ltcp.OG and len(temp) > 0

    def _generate_closed_cluster_sets(self, clusters: List[ClusterClass], LTCP_next: List[LTCPPattern]) -> List[ClusterSet]:
        """
        Generate cluster sets (non-empty subset of C), whose union-members are not already present as OG in LTCP_next
         (i.e., closed w.r.t current candidate OGs)
         Optionally prune by max)group_size
        """
        n = len(clusters)
        if n == 0:
            return []
        out: List[ClusterSet] = []
        # existing_OGs = {frozenset(p.OG) for p in LTCP_next}
        existing_OGs = [list(sorted(list(p.OG))) for p in LTCP_next]
        # Generate all non-empty subsets of clusters (except empty set)
        for r in range(1, len(clusters) + 1):
            for subset in combinations(clusters, r):
                cs = ClusterSet.from_clusters(t=clusters[0].time, clusters=subset)  # t will be reset by caller when adding
                if self.max_group_size is not None and len(cs.union_members) > self.max_group_size:
                    continue
                # print(list(sorted(list(cs.union_members))))
                flag = False
                for temp in existing_OGs:
                    if set(cs.union_members).issubset(set(temp)) or set(cs.union_members)== set(temp):
                        # if list(sorted(list(cs.union_members))) in existing_OGs:
                        flag = True
                        continue  # not closed w.r.t current candidate OGs
                if flag:
                    continue
                out.append(cs)
                # print(cs.union_members, existing_OGs)

        return out

    def _deduplicate(self, patterns: List[LTCPPattern]) -> List[LTCPPattern]:
        """Remove exact duplicates (same OG and same sequnence of cluster-set time indices"""
        # seen: Set[Tuple[FrozenSet[str], Tuple[int, ...]]] = set()
        seen = []
        out: List[LTCPPattern] = []
        for P in patterns:
            # key = (list(P.OG), list(sorted(cs.t for cs in P.TS)))
            # key = (list(P.OG), tuple(cs.t for cs in P.TS))
            key = (list(P.OG), list(P.timestamps))
            if key not in seen:
                # seen.add(key)
                seen.append(key)
                # seen.add(key)
                out.append(P)
        return out

    def discover_loose_traveling_companions_smart_and_fast_(self, objects: Dict[str, MovingObject] = None, mG: int = None,
                                            dG: int = None, fC: int = None, eps: float = None, minPts: int = None):
        """Run loose traveling companion pattern (LTCP) discovery across ordered time slots"""

        if objects is None:
            objects = self.objects
        if mG is None:
            mG = self.mG
        if dG is None:
            dG = self.dG
        if fC is None:
            fC = self.fC
        if eps is None:
            eps = self.eps
        if minPts is None:
            minPts = self.min_pts

        snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(objects)
        sorted_times = sorted(snapshots.keys())
        LTCP_closed: List[LTCPPattern] = []
        LTCP_active: List[LTCPPattern] = []  # current active loose traveling companions

        for t in sorted_times:
            snapshot = snapshots[t]
            clusters = snapshot.cluster_snapshot(snapshot, eps=eps, minPts=minPts)  # clusters at t
            LTCP_next: List[LTCPPattern] = []
            # 1) Extend existing candidates
            for ltcp in LTCP_active:
                ltcp.extended = False
                temp_cluster_sets = []
                temp_OG = []
                while True:
                    o = random.choice(list(ltcp.object_group))
                    cluster = clusters[snapshot.clusters_map[o]]
                    intersection = ltcp.OG & set(cluster.members)
                    if intersection:
                        if set(cluster.members).issubset(ltcp.OG):
                            temp_cluster_sets.append(cluster)
                            temp_OG.extend(cluster.members)
                        else:
                            break

                    if set(temp_OG) == ltcp.OG:
                        ltcp.extended = True
                        cs = ClusterSet.from_clusters(t, temp_cluster_sets)
                        ltcp_next = LTCPPattern(object_group=ltcp.OG, cluster_sequences=ltcp.TS + [cs])
                        ltcp_next.valid = self._is_valid(ltcp_next)
                        ltcp.extended = True
                        LTCP_next.append(ltcp_next)
                        break

                if not ltcp.extended:
                    # Not extended: if valid and cannot extend further, consider as closed now
                    if ltcp.valid and not self._exists_sup_extension(ltcp, list(clusters.values()), t):
                        LTCP_closed.append(ltcp)

            # Create new candidates from closed subset-collection at t
            new_candidates = self._generate_closed_cluster_sets(list(clusters.values()), LTCP_next)

            for cs in new_candidates:
                if len(cs.union_members) >= self.mG:
                    new_ltcp = LTCPPattern(object_group=cs.union_members, cluster_sequences=[cs], valid=False)
                    LTCP_next.append(new_ltcp)

            LTCP_active = list(set(LTCP_next))

            # Finalize: any remaining valid, non-extendable candidates are closed

        for ltcp in LTCP_active:
            if ltcp.valid:
                LTCP_closed.append(ltcp)
        return self._deduplicate(LTCP_closed)


class WeaklyConsistentLooseTravelingCompanionMiner(LooseTravelingCompanionMiner):
    """Discover closed weakly consistent Loose Traveling Companion Patterns (WCLTCP) per Naserian et al. (2016).

    New parameters
    ----------
    lC: int
        Max allowed gap in slots between consecutive cluster-sets in a pattern.
    """

    def __init__(self, objects: Dict[str, MovingObject], mG: int = 2, dG: int = 2, fC: int = 5, lC: int = 5,
                 eps: float = 20, minPts: int = 2, max_group_size: int = 100):
        super().__init__(objects=objects, mG=mG, dG=dG, fC=fC, eps=eps, minPts=minPts, max_group_size=max_group_size)
        self.lC = lC

    def discover_weakly_consistent_loose_traveling_companions(self, objects: Dict[str, MovingObject] = None,
                                                              mG: int = None, dG: int = None, fC: int = None,
                                                              lC: int = None, eps: float = None, minPts: int = None):

        if objects is None:
            objects = self.objects
        if mG is None:
            mG = self.mG
        if dG is None:
            dG = self.dG
        if fC is None:
            fC = self.fC
        if lC is None:
            lC = self.lC
        if eps is None:
            eps = self.eps
        if minPts is None:
            minPts = self.minPts

        snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(objects)
        sorted_times = sorted(snapshots.keys())
        LTCP_closed: List[LTCPPattern] = []

        LTCP_active: List[LTCPPattern] = []  # current active loose traveling companions

        for t in sorted_times:

            snapshot = snapshots[t]
            clusters = snapshot.cluster_snapshot(snapshot, eps=eps, minPts=minPts)  # clusters at t
            LTCP_next: List[LTCPPattern] = []
            # 1) Extend existing candidates
            for ltcp in LTCP_active:
                ltcp.extended = False
                temp_cluster_sets = []

                for cid, cluster in clusters.items():
                    # Find all clusters that are subsets of the candidate's object group
                    if set(cluster.members).issubset(ltcp.OG):
                        temp_cluster_sets.append(cluster)

                # Check if the union of these clusters equals the candidate's object group
                if temp_cluster_sets:
                    object_union = set()
                    for temp in temp_cluster_sets:
                        object_union.update(set(temp.members))

                    if object_union == ltcp.OG and len(temp_cluster_sets) > 0:
                        cs = ClusterSet.from_clusters(t, temp_cluster_sets)
                        ltcp_next = LTCPPattern(object_group=ltcp.OG, cluster_sequences=ltcp.TS + [cs])
                        ltcp_next.valid = self._is_valid(ltcp_next)
                        ltcp.extended = True
                        LTCP_next.append(ltcp_next)

                if not ltcp.extended:
                    # Not extended: check if gap allowed
                    if ltcp.TS:
                        gap = t - ltcp.get_end_time()
                        if gap <= self.lC:
                            # Carry forward without extension
                            LTCP_next.append(ltcp)
                            continue
                    # Otherwise, if valid and not extendable within lC, close it
                    if ltcp.valid and not self._exists_sup_extension(ltcp, list(clusters.values()), t):
                        LTCP_closed.append(ltcp)

            # Create new candidates from closed subset-collection at t
            for cs in self._generate_closed_cluster_sets(list(clusters.values()), LTCP_next):
                if len(cs.union_members) >= self.mG:
                    cs = ClusterSet(t=t, clusters=cs.clusters, union_members=cs.union_members)  # fix time
                    new_ltcp = LTCPPattern(object_group=cs.union_members, cluster_sequences=[cs], valid=False)
                    LTCP_next.append(new_ltcp)

            # Ensure times are set when extending carried-forward candidates
            # (For extended candidates above, time is already set via _try_extend_candidate)
            for i, ltcp in enumerate(LTCP_next):
                if ltcp.TS and ltcp.TS[-1].t == -1:
                    LTCP_next[i].TS[-1] = ClusterSet(t=t, clusters=ltcp.TS[-1].clusters,
                                                     union_members=ltcp.TS[-1].union_members)

            LTCP_active = LTCP_next

            # Finalize: any remaining valid, non-extendable candidates are closed
        for ltcp in LTCP_active:
            if ltcp.valid:
                LTCP_closed.append(ltcp)

        return self._deduplicate(LTCP_closed)
