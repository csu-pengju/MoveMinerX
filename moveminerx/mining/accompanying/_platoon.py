#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _platoon4.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/8/11 16:33
from collections import Counter
from typing import Dict, List, Tuple
from moveminerx.mining._basic_class import MovingObject, Snapshot, SnapshotBuilder, ClusterClass, Platoon, PrefixTable, PTEntry, PrefixListEntry

import sys

sys.setrecursionlimit(100000)


class PlatoonMiner:
    """Main class for mining platoon motion_mining from trajectory data"""

    def __init__(self, objects: Dict[str, MovingObject], min_o: int = 2, min_t: int = 2, min_c: int = 2,
                 eps: float = 20, minPts: int = 2, clustering_metric='precomputed'):
        """
        Initialize the platoon miner with thresholds
        Parameters:
            objects:
            min_o: minimum number of objects in a platoon
            min_t: minimum number of timestamps
            min_c: minimum consecutive timestamps in each segment
            eps:
            minPts:
        """
        self.objects = objects
        self.min_o = min_o  # 最小时间持续长度
        self.min_t = min_t  #
        self.min_c = min_c
        self.eps = eps
        self.minPts = minPts
        self.clustering_metric = clustering_metric
        # self.snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(self.objects)

    def run(self):

        snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(self.objects)
        # Clustering at each snapshot
        # derive OS (objects set) from clusters
        all_objs = set()
        timestamps = sorted(snapshots.keys())
        total_CDB_size = 0
        CDB = []
        for snapshot in snapshots.values():
            clusters = snapshot.cluster_snapshot(snapshot, minPts=self.minPts, eps=self.eps)

            for cluster in clusters.values():
                # print(snapshot.time, cluster.members)
                all_objs.update(cluster.members)
                total_CDB_size += 1
                CDB.append(cluster)
        # define a deterministic OS order (lexicographic)
        OS_order = sorted(all_objs)
        found_patterns, found_index = self._platoon_miner(CDB, snapshots, tuple(), timestamps, total_CDB_size,
                                                          min_o=self.min_o, min_t=self.min_t, min_c=self.min_c, depth=0,
                                                          OS_order=OS_order)
        return found_patterns

    def _platoon_miner(self, CDB: List[ClusterClass], snapshots: Dict[int, Snapshot], suffix_X: Tuple, TS_all: List,
                       total_CDB_size: int,
                       min_o: int, min_t: int, min_c: int, depth: int, OS_order: List = None,
                       found_patterns: List[Platoon] = None, found_index: Dict = None, s: int = 2):
        """
        Recursive PlatoonMiner. This is a direct-style implementation that keeps:
            - found_patterns: list of returned closed platoons
            - found_index: dict mapping tuple(Tmax_after_extract) -> list of (objectset, N) for subset checking

        For simplicity, CDB parameter is a list of Cluster-like entries where cluster.objects is a list of objects (ordered).
        suffix_X is the current suffix (tuple). At the root call, suffix_X = () and CDB is full temporal object cluster DB (scan once).
        """
        if found_patterns is None:
            found_patterns = []
        if found_index is None:
            found_index = dict()
        # Step 1: build PT for suffix_X using Insert-Table (Algorithm 2)
        PT = PrefixTable(suffix_X)
        PT = self._insert_table(PT, snapshots)

        # Step 2: process each object in the prefix table.
        # For each object o in PT, compute Sminc-con(Tmax) and Ncon, and detect common prefix objects
        CP = set()  # Common prefix
        RO = set()  # Objects to remove
        N_current = sum(cl.occ for cl in CDB)
        for o in list(PT.order):
            # print('*'*20, o, PT.table.get(o).Tmax)
            entry = PT.table.get(o)
            # Compute Sminc-con and Ncon
            S_min_c_con, Ncon = self._extract_local_timestamps(entry.Tmax, min_c)
            # print(S_min_c_con, Ncon)
            entry.Sminc_con = S_min_c_con
            entry.Ncon = Ncon

            # Pruning:
            # 1. Frequence-consecutive pruning: we require |Sminc-con(Tmax)| >= mint (paper suggests testing jSminc−con(Tmax)j >= mint)
            # But note paper uses |T| >= mint and decomposition: here Sminc-con is the set of consecutive segments; the paper
            # counts total timestamps in Tmax and also Sminc-con segments. We'll follow the paper: if total # timestamps (after extracting)
            # less than mint, prune.
            # 2. Common prefix pruning detection
            # If Ncon (occurrence of locally-consecutive timestamps) equals N (which is number of occurrences of X),
            # but at root X=(), the 'N' passed in paper equals |CDB| initially; we'll approximate by checking equality of Ncon and total_CDB_size
            # More accurate behaviour requires tracking N (occurrences) per candidate; we use N = number of clusters in current CDB (sum occ)

            total_T_count = sum(len(seg) for seg in S_min_c_con)
            # print(f"Prune {o}: total_T_count={total_T_count} < min_t={min_t}")
            if Ncon == N_current:
                CP.add(o)
            elif total_T_count < min_t:
                RO.add(o)
        # Add common prefix to remove objects
        RO.update(CP)

        # Step 3: Handle common prefix case
        if CP:
            # build objectset (CP ∪ X), determine Tmax as combined timestamps — in PT we can take any o in CP entry.Tmax then extract Sminc
            # For simplicity, use the first CP object to get the Sminc-con segments (they should be identical by Ncon equality).
            sample_o = list(CP)[0]
            sample_entry = PT.table[sample_o]
            # reconstruct T as flattened union (paper uses Tmax (before extraction) but outputs T being Sminc-con(Tmax) flattened)
            T_merged = []
            for seg in sample_entry.Sminc_con:
                T_merged.extend(seg)

            # If Common prefixes (CP) non-empty and CP ∪ X satisfies min_o, we can output closed platoon  CP ∪ X : T : N
            if len(CP) + len(suffix_X) >= min_o and len(T_merged) >= min_t:
                # N is N_current
                times = PT.table[list(CP)[0]].Tmax
                new_platoon = Platoon(objectset=tuple(sorted(set(CP).union(set(suffix_X)))),
                                      timestamps=times, N=N_current)
                # print('dsd', suffix_X, CP, new_platoon.objectset)
                is_closed = all(new_platoon.is_closed(p) for p in found_patterns)
                if is_closed:
                    found_patterns.append(new_platoon)
                    # update index for subset checking
                    found_index.setdefault(tuple(T_merged), []).append((new_platoon.objectset, new_platoon.N))
        else:
            # No common prefix, check if current candidate is a closed platoon
            if len(suffix_X) >= self.min_o and s == 2:
                times = PT.table[list(suffix_X)[0]].Tmax
                new_platoon = Platoon(objectset=suffix_X, timestamps=times, N=N_current)
                # Check if this platoon is closed
                is_closed = all(new_platoon.is_closed(p) for p in found_patterns)
                if is_closed:
                    found_patterns.append(new_platoon)

        # Step 4: Remove objects that don't meet criteria
        for o in RO:
            if o in PT.table:
                PT.order.remove(o)
                del PT.table[o]

        # Step 5 Recursively process each remaining object o in reverse order
        # subtree substitution: replace subtree of X with subtree of (CP ∪ X)
        # For simplicity we will proceed by setting suffix_X' = (CP_first, ) + suffix_X and continue exploring children accordingly.
        # Implementation detail: paper suggests substituting tree; here we will still explore children but skip branches that don't contain CP.
        # So we will prune any object o' in PT.order that does not lead to CP-containing descendants.
        # Implement conservative pruning: keep only objects that are in CP or succeed CP in order.

        for o in reversed(PT.order):
            if OS_order:
                try:
                    i = OS_order.index(o) + 1
                except ValueError:
                    i = None
            else:
                i = None

            if i is not None:
                if (i + len(suffix_X)) < min_o:
                    # pruning
                    continue

            # Create new suffix (child objectset X' = {o} ∪ X)
            Xprime = tuple([o] + list(suffix_X))

            # Get the prefix list for this object
            plo_map = PT.table[o].PLo
            Tmax_candidate = PT.table[o].Tmax
            S_min_c_con, Ncon = self._extract_local_timestamps(Tmax_candidate, min_c)

            # flatten timestamps as T and obtain valid timestamps
            T_flat = []
            for seg in S_min_c_con:
                T_flat.extend(seg)

            # Subset checking
            s_check = self._subset_checking(tuple(Xprime), T_flat, Ncon, found_index)
            if s_check == 0:
                continue

            # construct child cluster dataset and snapshots
            child_cdb, new_snapshots = self._build_child_dataset(plo_map=plo_map, S_min_c_con=S_min_c_con, o=o)

            if new_snapshots or len(Xprime) >= min_o:

                self._platoon_miner(child_cdb, new_snapshots, Xprime, TS_all,
                                    total_CDB_size,
                                    min_o, min_t, min_c, depth + 1,
                                    OS_order=OS_order,
                                    found_patterns=found_patterns,
                                    found_index=found_index, s=s_check)
                # self._platoon_miner()

        return found_patterns, found_index

    def _build_child_dataset(self, plo_map, S_min_c_con, o):

        valid_times_set = set(ts for seg in S_min_c_con for ts in seg)
        child_cdb = []
        new_snapshots: Dict[int, Snapshot] = {}
        for prefix, ple in plo_map.items():
            counter = Counter(ple.Tp)
            for t, cnt in counter.items():
                if t not in valid_times_set:
                    continue
                cluster_objs = list(prefix) + [o]
                cluster = ClusterClass(cid='-1', t=t, members=cluster_objs)
                child_cdb.append(cluster)
                if t in new_snapshots.keys():
                    new_snapshots[t].clusters.append(cluster)
                else:
                    new_snapshot = Snapshot(time=t)
                    new_snapshot.clusters = [cluster]
                    new_snapshots[t] = new_snapshot
        return child_cdb, new_snapshots

    def _subset_checking(self, candidate_O: Tuple, T, N, found_patterns_index: Dict):
        """
          Subset checking against previously found pattern set R.
          found_patterns_index: dict mapping timestamp_key -> list of (objectset_tuple, N)
             - timestamp_key must uniquely identify a timestamp multiset; here we use tuple(T) as key
          Returns:
           - 0 if there exists C' in R with O ⊂ O' and T == T' and N == N'  (prune)
           - 1 if there exists C' in R with O ⊂ O' and T == T' and N < N'   (not closed but descendants may have closed)
           - 2 otherwise (no superset found)
          """
        # create key
        key = tuple(T)
        if key not in found_patterns_index:
            return 2

        candidates = found_patterns_index[key]
        Oset = set(candidate_O)
        for Oprime, Nprime in candidates:
            Oprime_set = set(Oprime)
            if Oset.issubset(Oprime_set):
                if N == Nprime:
                    return 0
                else:
                    return 1
        return 2

    def _extract_local_timestamps(self, tmax_list: List, min_c: int):
        """
        Given Tmax (a list of timestamps, duplicates allowed, ordered by scanning CDB left-to-right),
        extract Sminc-con(Tmax): the set of maximal consecutive timestamp sequences each having length >= minc.
        Also return Ncon: the number of occurrences (counting duplicates) in these segments (as in paper).
        Implementation follows the scan-left-to-right logic from the paper's Algorithm 3.
        """
        if not tmax_list:
            return [], 0
        # remove None and ensure sorted by timestamp order if possible
        # tmax_list is expected to already be in time-encounter order
        segments = []
        Ncon = 0

        seg = [tmax_list[0]]
        # c counts occurrences (including duplicates) within current consecutive block
        c = 1
        for prev, cur in zip(tmax_list, tmax_list[1:]):
            if cur - prev == 1:
                seg.append(cur)
                c += 1
            else:
                # block ended; decide to push if length >= minc
                if len(seg) >= min_c:
                    segments.append(list(seg))
                    Ncon += c
                # reset
                seg = [cur]
                c = 1
            # finish last block
        if len(seg) >= min_c:
            segments.append(list(seg))
            Ncon += c

        return segments, Ncon

    def _insert_table(self, pt: PrefixTable, snapshots: Dict[int, Snapshot]) -> PrefixTable:
        """
        Build a prefix table fot the given suffix
        Each cluster C in CDB has objects in a (left-to-right) order. For each object o in cluster.objects:
            - merge its Tmax with C.timestamp
            - update PLo for all prefixes P of o (prefix: preceding objects in cluster.objects)
        Parameters:
             snapshots:
             pt:
        Returns:
            A dict of the constructed prefix table: {object: {'T_max': list of timestamps, 'N': count, 'prefixes': {...}}}
        """

        # For each cluster record
        for t, snapshot in snapshots.items():
            for cluster in snapshot.clusters:
                objs = sorted(cluster.members)
                # L = len(objs)
                # scan left-to-right as described in paper
                for idx, o in enumerate(objs):
                    # ensure entry
                    if o not in pt.table:
                        pt.table[o] = PTEntry()
                        pt.order.append(o)
                    # Merge Tmax with this timestamp
                    pt.table[o].Tmax.extend([t])

                    # Update PLo: prefixes are objects to the left of o
                    # prefix P we record the timestamp for the pair (P ∪ {o})
                    if idx > 0:
                        prefix = tuple(objs[:idx])
                        plo = pt.table[o].PLo.get(prefix)
                        if plo is None:
                            plo = PrefixListEntry()
                            pt.table[o].PLo[prefix] = plo
                        plo.Tp.extend([t])
                        plo.Np += 1

        return pt
