#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _swarm.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/8/6 20:55


from collections import defaultdict
from typing import Dict, Set, Tuple, List
from moveminerx.mining._basic_class import Snapshot, MovingObject, SnapshotBuilder, SwarmTreeNode, SwarmPattern


class SwarmMiner:
    def __init__(self, objects: Dict[str, MovingObject], min_o=2, min_t=3, eps: float = 5, minPts: int = 2,
                 theta: float = 1.0):
        """
        The implementation of ObjectGrowth algorithm  for mining closed swarms in paper 'Swarm: Mining Relaxed Temporal Moving Object Clusters'.
        Swarm pattern relaxes the consecutive time constraint in which is different from flock, convoy and moving cluster patterns.
        It requires time-closed and object-closed to make sure reducing the redundant swarm patterns.
        ObjectGrowth algorithm adopts a depth-first search order from one object to mostly objects.

        Parameters:
            - objects:
            - min_o: the minimum number of moving objects the swarms require.
            - min_t: the minimum span of timestamps the swarms require.
            - eps & minPts: parameters for DBSCAN clustering
            - theta: confidence threshold (default 1.0 for certain data)
        Returns:
            the detected closed swarm patterns
        """
        self.objects = objects
        self.min_o = min_o
        self.min_t = min_t
        self.eps = eps
        self.minPts = minPts
        self.theta = theta
        self.timestamps = []  # list of all timestamps
        self.object_ids = []  # list of all object IDs
        self.snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(self.objects)
        self.swarm_tree = None  # root of the swarm search tree
        # self.T_max = max(obj.trajectory.end_time for obj in self.objects.values())
        self.object_clusters = defaultdict(dict)  # object_id -> timestamp -> list of clusters 每个目标所属的簇
        self.probabilities = defaultdict(dict)  # object_id -> timestamp -> probability
        # self.min_time = min(obj.trajectory.start_time for obj in self.objects.values())
        # self._get_max_timeset(objectset=self.object_ids)

        self._snapshots_cluster(snapshots=self.snapshots, eps=self.eps, minPts=self.minPts)

    def run(self):

        # Start with empty objectset and all timestamps
        initial_timeset = set(self.timestamps)
        # Begin recursive mining and build swarm tree
        all_swarms, _ = self._object_growth(set(), initial_timeset, -1)
        # Filter to only include valid closed swarms
        closed_swarms = []
        seen = set()
        for objset, timeset in all_swarms:
            if len(objset) >= self.min_o and len(timeset) >= self.min_t:
                # Create a unique key for the swarm
                key = (frozenset(objset), frozenset(timeset))
                if key not in seen:
                    seen.add(key)
                    closed_swarms.append((objset, timeset))

        formatted_closed_swarms = self._format_mined_swarms(closed_swarms)
        return formatted_closed_swarms

    def add_probability(self, object_id: str, timestamp: int, probability: float):
        """Add probability information for an object at a timestamp"""
        self.probabilities[object_id][timestamp] = probability

    def _snapshots_cluster(self, snapshots: Dict[int, Snapshot], eps: float, minPts: int, metric='precomputed'):
        """Get timestamps """
        for t, snapshot in snapshots.items():
            clusters = snapshot.cluster_snapshot(snapshot, eps=eps, minPts=minPts, metric=metric)
            if t not in self.timestamps:
                self.timestamps.append(t)
                self.timestamps.sort()
            # update object_cluster mapping
            for cluster in clusters.values():
                for object_id in cluster.members:
                    if object_id not in self.object_ids:
                        self.object_ids.append(object_id)

                    if object_id not in self.object_clusters:
                        self.object_clusters[object_id] = {}

                    if t not in self.object_clusters[object_id]:
                        self.object_clusters[object_id][t] = []

                    self.object_clusters[object_id][t].append(cluster)

    def _get_max_timeset(self, objectset: Set[str], max_timeset: Set[int], new_obj: str) -> Set[int]:
        """
        Compute the maximal timeset T_max(O) for a given objectset O.
        Returns the set of timestamps where all objects in O are in the same cluster.
        """
        new_max_timeset = set()
        for t in max_timeset:
            # Check if new_obj is in any cluster that contains all of objectset at time t
            # 检查new_obj是否在objectset的所有对象共现的聚类中
            # Get intersection of clusters containing all objects in objectset at t
            # 情况1：objectset为空时，直接检查new_obj是否存在
            # Get all clusters at time t that contain the first object
            if not objectset:
                if self.object_clusters[new_obj].get(t, []):  # new_obj在t时刻有聚类
                    new_max_timeset.add(t)
                continue

            # 情况2：objectset非空时，检查共现聚类
            common_clusters = None
            for obj in objectset:
                obj_clusters = set(c.cid for c in self.object_clusters[obj][t])
                # print(t, 'obj_clusters', obj_clusters)
                if common_clusters is None:
                    common_clusters = obj_clusters
                else:
                    common_clusters.intersection_update(obj_clusters)
                if not common_clusters:
                    break

            if not common_clusters:
                continue
            # print(f'common_clusters: {common_clusters}')
            # Check if new_obj is in any of these common clusters
            new_obj_clusters = set(c.cid for c in self.object_clusters[new_obj].get(t, []))
            # print(t, 'new_objectset', new_objectset, 'new_obj_clusters', new_obj_clusters)
            if common_clusters.intersection(new_obj_clusters):
                new_max_timeset.add(t)

        return new_max_timeset

    def _apriori_prune(self, max_timeset: Set[int], min_t: float) -> bool:
        """
        Apriori pruning rule: prune if |T_max(O)| < min_t or f(O,T) < theta*min_t
        Returns True if should prune, False otherwise
        """
        # apriori性质.
        # 如果一个项集是频繁的，其所有子集也是频繁的。相反, 如果一个项集是非频繁的, 其所有超集无须再检查, 直接排除
        if len(max_timeset) < min_t:
            return True
        return False

    def _backward_prune(self, objectset: Set[str], max_timeset: Set[int], last_added: str) -> bool:
        """
        Backward pruning rule: check if there exists an object o' < last_added that is in the same cluster
        as objectset for all t in max_timeset.

        Parameters:
            objectset:
            max_timeset:
            last_added:
        Returns: True if should prune, False otherwise.
        """
        if not objectset or not max_timeset:
            return False

        # Get the index of the last added object
        try:
            last_idx = self.object_ids.index(last_added)
        except ValueError:
            return False

        # Check all objects before last_added
        # 检查编号更小的对象
        for i in range(last_idx):
            candidate_obj = self.object_ids[i]
            if candidate_obj in objectset:
                continue
            # Check if candidate_obj is in same cluster as objectset for all max_timeset
            valid = True

            for t in max_timeset:
                # Get clusters of objectset at time t
                clusters = None
                for obj in objectset:
                    obj_clusters = set(c.cid for c in self.object_clusters[obj][t])
                    if clusters is None:
                        clusters = obj_clusters
                    else:
                        clusters.intersection_update(obj_clusters)
                    if not clusters:
                        break
                if not clusters:
                    valid = False
                    break

                # Check if candidate_obj is in any of these clusters
                candidate_clusters = set(c.cid for c in self.object_clusters[candidate_obj].get(t, []))
                if not clusters.intersection(candidate_clusters):
                    valid = False
                    break
            if valid:
                return True  # should prune
        return False  # no need to prune

    def _forward_closure_check(self, objectset: Set[str], max_timeset: Set[int],
                               children_results: List[Tuple[Set[str], Set[int]]]) -> bool:

        """
        Forward closure checking: check if any child has the same timeset size

        Returns:
            True if current objectset is closed, False otherwise
        """
        for child_objset, child_timeset in children_results:
            if len(child_timeset) == len(max_timeset):
                return False  # Not closed
        return True  # Closed

    def _object_growth(self, objectset: Set[str], max_timeset: Set[int], last_added_idx: int,
                       parent_node: SwarmTreeNode = None) -> Tuple[List[Tuple[Set[str], Set[int]]], SwarmTreeNode]:
        """
        Recursive ObjectGrowth algorithm to mine closed swarms.
        Parameters:
            objectset:
            max_timeset:
            last_added_idx:
            parent_node:
        Returns:
            - list of (objectset, timeset, confidence) tuples representing closed swarms
            - the current node in the swarm tree
        """
        results = []
        current_node = SwarmTreeNode(objectset.copy(), max_timeset.copy())

        if parent_node:
            parent_node.add_child(current_node)
        else:
            self.swarm_tree = current_node  # Set as root if no parent

        # Apply Apriori pruning to filter impossible swarm nodes and their child nodes,
        # i.e., prune objectsets with Tmax(0)<min_t 提前终止不可能满足 min_t 约束的搜索路径
        if self._apriori_prune(max_timeset=max_timeset, min_t=self.min_t):
            return results, current_node

        # Apply Backward pruning 避免生成冗余的Closed Swarm候选。
        # 若存在对象 o'(编号小于当前最后添加的对象),
        # 使得 o' 在所有 T_max(O) 时间戳上与 O 同属一个聚类,则 O 的扩展无法生成Closed Swarm (因为 O ∪ {o'} 会覆盖相同时间集)。
        last_added = self.object_ids[last_added_idx] if last_added_idx >= 0 else None

        # print(last_added, self._backward_prune(objectset, max_timeset, last_added))
        # print(results, current_node, 'objectset', objectset, 'max_timeset', max_timeset)
        if last_added and self._backward_prune(objectset, max_timeset, last_added):
            return results, current_node
        # print(objectset, max_timeset, 'dsdsdsdsdsd')
        children_results = []
        # Explore all possible extensions with objects after last_added
        for i in range(last_added_idx + 1, len(self.object_ids)):
            new_obj = self.object_ids[i]  # 获取新对象
            new_objectset = objectset.copy()  # 生成超集
            new_objectset.add(new_obj)
            # Calculate new_max_timeset(only need to check timestamps in current max_timeset)
            new_max_timeset = self._get_max_timeset(objectset=objectset, max_timeset=max_timeset, new_obj=new_obj)

            # print(f'new_objectset: {new_objectset}, new_max_timeset: {new_max_timeset}')
            # Recursively process the new objectset
            child_swarms, child_node = self._object_growth(new_objectset, new_max_timeset, i, current_node)
            # Store this child's results for closure checking
            children_results.extend(child_swarms)
            # children_results.append((new_objectset, new_max_timeset))

        # Forward closure checking
        if (self._forward_closure_check(objectset, max_timeset, children_results)) and len(objectset) >= self.min_o:
            results.append((objectset.copy(), max_timeset.copy()))

        results.extend(children_results)
        return results, current_node



    def _format_mined_swarms(self, swarms: List[Tuple[Set, Set]]) -> Dict[int, SwarmPattern]:
        S_final_patterns: Dict[int, SwarmPattern] = {}
        pattern_counter = 0
        for objectset, max_timeset in swarms:
            pattern_counter += 1
            s = SwarmPattern(objectset=objectset, max_timeset=max_timeset)
            s.extend_swarm_multiple_times(objectset=objectset, max_timeset=max_timeset)
            s.pattern_counter = pattern_counter
            S_final_patterns[pattern_counter] = s
        return S_final_patterns




