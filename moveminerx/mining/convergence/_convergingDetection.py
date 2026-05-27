#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _convergenceDivergencePatternDetection.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/10/21 9:10
import itertools
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN

# from basic_Class import TrajectoryLoader
from moveminerx.mining.convergence._basic_class import MovingObject, Snapshot, SnapshotBuilder, ConvergingTree, ConvergingTreeNode, \
    Cluster, ClusterContainmentMatch, ConvergingPattern, TrajectoryLoader


class ConvergingPatternMiner:
    def __init__(self, objects: Dict[str, MovingObject], k_t: int = 2, k_m: int = 2, k_p: int = 2, eps: float = 20,
                 minPts: int = 2, m_sig: int = 1024, k_sig: int = 4):
        """
        汇聚模式挖掘器基类
        Initialize the miner with parameters
        Parameters:
            objects:
            k_t: the lifetime threshold of a converging
            k_m: the support threshold of a converging
            k_p: the lifetime threshold of a participator
            m_sig:
            k_sig:
            eps: the radius threshold used for DBSCAN used for moving object clustering at the timestamp.
            minPts: the minimum neighbors(moving objects) of a moving object to be a core object of DBSCAN.
        """
        self.objects = objects
        self.k_p = k_p
        self.k_t = k_t
        self.k_m = k_m
        self.eps = eps
        self.minPts = minPts
        self.m_sig = m_sig
        self.k_sig = k_sig
        self.snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(self.objects)
        self.clustering_metric = 'precomputed'
        self.converging_trees: List[ConvergingTree] = []
        self.matches: List[ClusterContainmentMatch] = []
        self.pattern_counter = 0

    def discovery_converging_patterns(self, end_time: int=None):
        # 在这里放参数也比较好, 修改比较方便

        sorted_times = sorted(self.snapshots.keys())
        if not end_time:
            end_time = sorted_times[-1]
        if len(sorted_times) < 2:
            return []
        clusters_db: Dict[int, Dict[str, Cluster]] = {}

        for idx, t in enumerate(sorted_times[:end_time]):
            # Phase 1: 快照聚类发现
            clusters = self.snapshots[t].cluster_snapshot(snapshot=self.snapshots[t], eps=self.eps, minPts=self.minPts,
                                                          metric=self.clustering_metric)
            clusters_db[t] = clusters
            # G_next: Dict[int, ConvergingTree] = {}
        # Phase 2: 聚类包含连接 (CCJ) 使用Nested-loops based CCJ (NLCCJ)
        R_DB: Dict[int, List[Tuple[Cluster, Cluster]]] = {}
        # print("\nPhase 2: Cluster Containment Join (NLCCJ)")
        for i in range(1, len(sorted_times[:end_time])):
            t_prev, t_curr = sorted_times[i - 1], sorted_times[i]
            matches = self._nlccj(clusters_db[t_prev], clusters_db[t_curr])
            R_DB[t_curr] = matches
            # print(f"  t={t_prev} -> t={t_curr}: Found {len(matches)} matches.")
        # Phase 3: 汇聚检测
        # Building ConvergingTree 阶段
        # print('R_DB', R_DB)
        self.build_converging_tree(end_time=end_time, R_DB=R_DB)
        # print('converging_trees', self.converging_trees)
        patterns = []
        # 验证和过滤
        for tree in self.converging_trees:
            pattern = ConvergingPattern(tree, self.k_t, self.k_m, self.k_p)
            if pattern.is_valid():
                patterns.append(pattern)

        return patterns

    def _nlccj(self, clusters_prev: Dict[str, Cluster], clusters_curr: Dict[str, Cluster]) -> List[Tuple[Cluster, Cluster]]:
        """
        Phase 2: 嵌套循环聚类包含连接 (NLCCJ) [cite: 659]。
        :param clusters_prev:
        :param clusters_curr:
        :return: t_{i-1} 和 t_i 之间的所有匹配 [(q, s), ...]
        """
        matches = []
        # 复杂度 O(|C_{i-1}| * |C_i|) [cite: 660]
        for q in clusters_prev.values():
            for s in clusters_curr.values():
                if q.is_contained_by(s):
                    matches.append((q, s))
                    self.matches.append(ClusterContainmentMatch(sub_cluster=q, super_cluster=s))
        return matches

    def build_converging_tree(self, end_time, R_DB: Dict[int, List[Tuple[Cluster, Cluster]]]):
        """构建汇聚树
        Parameters:
            clusters_db:时间快照簇
            R_DB: 时间戳之间的簇包含匹配
        Return:

        """
        # print('Building converging trees......')
        # 按时间排序快照
        sorted_timestamps = sorted(self.snapshots.keys())
        # 为每个匹配构建树结构
        cluster_nodes = {}  # cluster_id -> TreeNode

        for i in range(1, len(sorted_timestamps[:end_time])):
            t_curr = sorted_timestamps[i]
            # 获取这两个时间戳之间的匹配
            current_matches = R_DB.get(t_curr, [])
            for sub_cluster, super_cluster in current_matches:
                sub_node = cluster_nodes.get(sub_cluster.cid)
                super_node = cluster_nodes.get(super_cluster.cid)
                # print('sub_cluster', sub_cluster, 'super_cluster', super_cluster)
                if sub_node is None:
                    sub_node = ConvergingTreeNode(sub_cluster)
                if super_node is None:
                    super_node = ConvergingTreeNode(super_cluster)
                super_node.add_child(sub_node)
                cluster_nodes[sub_cluster.cid] = sub_node
                cluster_nodes[super_cluster.cid] = super_node

        root_nodes = [node for node in cluster_nodes.values() if node.parent is None]
        # print('root nodes', root_nodes)
        # 创建汇聚树
        # 创建汇聚树
        for root in root_nodes:
            converging_tree = ConvergingTree(root)
            self.converging_trees.append(converging_tree)

    def _find_closed_converging(self, all_trees: List[ConvergingTree], k_t: int, k_m: int, k_p: int):
        """
        Phase 3: 验证有效模式并过滤出闭合汇聚模式 (Closed Converging) [cite: 590]。
        （闭合模式简化为：满足所有阈值约束且在根节点对象集和高度上最大化的树）
        """
        valid_convergings = [tree for tree in all_trees if tree.is_valid_converging(k_t, k_m, k_p)]
        if not valid_convergings:
            return []
        # 找出高度和对象集大小最大的模式
        max_height = max(t.height for t in valid_convergings)
        max_obj_set_size = max(len(t.root.cluster.members) for t in valid_convergings if t.height == max_height)
        closed_convergings = [
            t for t in valid_convergings
            if t.height == max_height and len(t.root.cluster.members) == max_obj_set_size
        ]
        # 基于根节点对象集合去重
        unique_closed = []
        seen_roots = set()
        for t in closed_convergings:
            root_key = tuple(sorted(list(t.root.cluster.members)))
            if root_key not in seen_roots:
                unique_closed.append(t)
                seen_roots.add(root_key)

        return 0

    def _pattern_counter(self):
        self.pattern_counter += 1
        return self.pattern_counter


# if __name__ == "__main__":
#     folder = r'D:\Projects\MovementPatternDetectionToolkit\generalCompanionPatternMining\simulated_Ex\data'
#     datasets = ['D1', 'D2', 'D3', 'D4']
#     for dataset in datasets[0:]:
#         objects = TrajectoryLoader.load_from_shp(rf'..\simulated Ex\data\{dataset}.shp')
#
#         convergenceTreeMiner = ConvergingPatternMiner(objects=objects, k_t=3, k_m=3, k_p=2, eps=200, minPts=2)
#
#         convergences = convergenceTreeMiner.discovery_converging_patterns()
#         print(f'converging-based algorithm for convergences : {convergences}')
#         break
