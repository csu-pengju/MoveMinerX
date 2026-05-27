#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _moving_cluster.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/7/25 11:30
import random

import numpy as np
from sklearn.cluster import DBSCAN
from moveminerx.mining._basic_class import MovingObject, Snapshot, SnapshotBuilder, MovingClusterPattern, ClusterClass
from typing import List, Tuple, Dict
from collections import defaultdict


class MCAlgorithm:

    def __init__(self, objects: Dict[str, MovingObject], k: int, theta: float, eps: float, minPts: int, alpha=0.05,
                 clustering_metric='precomputed'):
        """
        The python completion of On Discovering Moving Clusters in Spatio-temporal Data, Kalnis et al. 2005.
        Moving cluster pattern detection.

        :param objects:
        :param k: the least timestamps of a valid moving cluster.
        :param theta: the minimum intersection threshold of the moving objects in the clusters between two consecutive timestamps
        :param eps: the radius threshold used for DBSCAN used for moving object clustering at the timestamp.
        :param minPts: the minimum neighbors(moving objects) of a moving object to be a core object of DBSCAN.
        :param alpha: 用户设定的误差阈值比例（建议在 0.05 ~ 0.2）
        :return:
        """

        self.objects = objects
        self.k = k  # 最小时间持续长度
        self.theta = theta
        self.eps = eps
        self.minPts = minPts
        self.snapshots: Dict[int, Snapshot] = SnapshotBuilder.generate_snapshots(self.objects)
        self.moving_clusters: List[Tuple[int, MovingClusterPattern]] = []
        self.clustering_metric = clustering_metric
        self.method = 'MC1'
        self.alpha = alpha

    def run(self):
        if self.method == 'MC1':
            return list(self.MC1())
        elif self.method == 'MC2':
            return list(self.MC2())
        elif self.method == 'MC3':
            return list(self.MC3())
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def MC1(self):
        sorted_times = sorted(self.snapshots.keys())
        pattern_counter = 0
        G_active_patterns: Dict[int, MovingClusterPattern] = {}  # 当前活跃的移动簇序列，每个元素为 List[Cluster]
        G_final_patterns: Dict[int, MovingClusterPattern] = {}  #
        for idx, t in enumerate(sorted_times):
            clusters = self.snapshots[t].cluster_snapshot(snapshot=self.snapshots[t], eps=self.eps, minPts=self.minPts,
                                                          metric=self.clustering_metric)
            # print(idx, t, [c.members for c in clusters.values()])
            G_next: Dict[int, MovingClusterPattern] = {}
            for mc_id, g in G_active_patterns.items():
                g.extended = False

            for cluster_id, c in clusters.items():
                # initialize the assigment of the cluster

                matched = False
                # traverse current existing moving clusters
                for mc_id, g in G_active_patterns.items():
                    # 获取该模式中上一个时刻的对象
                    g_last_objects = g.time_to_objects[sorted_times[idx - 1]]
                    # judge if the cluster c to be assigned and the existing moving cluster of the last timestamp can form a valid moving cluster
                    inter = len(g_last_objects & set(c.members))
                    union = len(g_last_objects | set(c.members))
                    # 如果该簇可以加入
                    if union > 0 and (inter / union) >= self.theta:
                        # extend G next moving clusters
                        g.extend_pattern(t, set(c.members))
                        g.extended = True
                        matched = True
                        #  update the next moving cluster set
                        G_next[mc_id] = g
                # if existing moving clusters cannot be extended by  clusters in current timestamp,
                # create new moving clusters and add them into next moving cluster set
                if not matched:
                    # 新建一个moving cluster
                    mc = MovingClusterPattern()
                    mc.extend_pattern(t, set(c.members))
                    pattern_counter += 1
                    mc.pattern_counter = pattern_counter
                    G_next[pattern_counter] = mc

            for mc_id, g in list(G_active_patterns.items()):

                # 输出或者删除不能被扩展的moving cluster
                if not g.extended:
                    if g.lifetime > self.k:
                        G_final_patterns[mc_id] = g
                    else:
                        del G_active_patterns[mc_id]

            G_active_patterns = G_next

        # 收尾: 剩余活跃模式判断是否满足条件
        for mc_id, g in G_active_patterns.items():
            if g.lifetime > self.k:
                G_final_patterns[mc_id] = g
        return G_final_patterns

    def MC2(self):
        sorted_times = sorted(self.snapshots.keys())
        pattern_counter = 0
        G_active_patterns: Dict[int, MovingClusterPattern] = {}  # 当前活跃的移动簇序列，每个元素为 List[Cluster]
        G_final_patterns: Dict[int, MovingClusterPattern] = {}  #
        for idx, t in enumerate(sorted_times):
            clusters = self.snapshots[t].cluster_snapshot(snapshot=self.snapshots[t], eps=self.eps, minPts=self.minPts,
                                                          metric=self.clustering_metric)
            G_next: Dict[int, MovingClusterPattern] = {}
            # for mc_id, g in G_active_patterns.items():
            #     g.extended = False
            # traverse current existing moving clusters
            for mc_id, g in G_active_patterns.items():
                g.extended = False
                # if g.removed:
                #     continue
                # 获取该模式中上一个时刻的对象
                g_last_t_objects = g.time_to_objects[sorted_times[idx - 1]]
                k = (1 - self.theta) * len(g_last_t_objects)
                while k > 0:
                    o_j = random.choice(list(g_last_t_objects))
                    c_res = self.find_object_in_clusters(o_j, clusters)
                    if not c_res[0]:
                        k = k - 1
                    else:
                        c = c_res[1]
                        inter = len(g_last_t_objects & set(c.members))
                        union = len(g_last_t_objects | set(c.members))
                        if union > 0 and inter / union >= self.theta:
                            g.extended = True
                            g.extend_pattern(t, set(c.members))
                            G_next[mc_id] = g
                            c.assigned = True
                            break
                        # 这里是否应有, 但是论文里面有, 思考一下为什么
                        # k = k - len(set(g_last_t_objects) - set(c.members))
                # if existing moving clusters cannot be extended by  clusters in current timestamp,
                # create new moving clusters and add them into next moving cluster set
                if not g.extended:
                    if g.lifetime >= self.k:
                        G_final_patterns[mc_id] = g
                    else:
                        g.removed = True
                        # del G_active_patterns[mc_id]

            for cid, c in clusters.items():
                if not c.assigned:
                    # 新建一个moving cluster
                    mc = MovingClusterPattern()
                    mc.extend_pattern(t, set(c.members))
                    pattern_counter += 1
                    mc.pattern_counter = pattern_counter
                    G_next[pattern_counter] = mc
            G_active_patterns = G_next

            # 收尾: 剩余活跃模式判断是否满足条件
        for mc_id, g in G_active_patterns.items():
            if g.lifetime > self.k:
                G_final_patterns[mc_id] = g
        return G_final_patterns

    def MC3(self):
        sorted_times = sorted(self.snapshots.keys())
        pattern_counter = 0
        G_active_patterns: Dict[int, MovingClusterPattern] = {}  # 当前活跃的移动簇序列，每个元素为 List[Cluster]
        G_final_patterns: Dict[int, MovingClusterPattern] = {}  #
        timer = 0
        period = 1
        for idx, t in enumerate(sorted_times):
            if timer < period:
                # 该方法背后的思想是移动对象不会频繁变换其之间的空间关系
                # 近似映射已有簇
                Si_objects = self.snapshots[t].points.values()
                # print('Si_objects', Si_objects)
                cluster_map = {}
                for mc_id, g in G_active_patterns.items():

                    g_last_t_objects = g.time_to_objects[sorted_times[idx - 1]]
                    # 构造一个hash map {odi: mc_id}
                    for oid in g_last_t_objects:
                        # 将该时刻所有的对象赋值给
                        cluster_map[oid] = mc_id

                approx_clusters = defaultdict(set)  # Dict[int, Cluster]
                # approx_clusters: Dict[int, Cluster] = {}
                # 对于当前时间快照, 如果目标在hash map
                for p in Si_objects:
                    if p.oid in cluster_map:
                        approx_clusters[id(cluster_map[p.oid])].add(p.oid)

                # L1 是此刻移动对象没有重叠的簇集合
                L1 = []
                # L1: Dict[int, Cluster] = {}
                for c_id, obj_ids in approx_clusters.items():
                    if len(obj_ids) >= self.minPts:
                        c = ClusterClass(cid='-1', t=t, members=list(obj_ids))

                        L1.append(c)
                        # L1[c_id] = c

                # 当前时刻不属于任何上一个时刻簇的移动对象, 对其进行聚类得到簇
                remaining = [p for p in Si_objects if all(p.oid not in c.members for c in L1)]
                L2 = self.dbscan(remaining, t)
                # L是所有对象进行了聚类或者近似聚类得到的簇结果
                L = L1 + list(L2.values())
                # 将其转为Dict[int, Cluster]格式, 相当于MC1和MC2中的clusters
                L = {cid: c for cid, c in enumerate(L)}
                timer += 1

            else:
                # 精确聚类
                L = self.snapshots[t].cluster_snapshot(snapshot=self.snapshots[t], eps=self.eps, minPts=self.minPts,
                                                       metric=self.clustering_metric)
                # === 动态调整 period（模仿 TCP/IP）===
                delta_clusters = abs(len(L) - len(G_active_patterns))
                # 根据 α 动态调整 period
                # 当某一时刻MC3执行精确聚类 (即 timer >= period)，我们对比当前与上一时刻的 moving cluster数量变化,
                # 如果变化超过一定比例 α,说明近似误差高,则将period 减半; 否则将 period递增。
                if delta_clusters > self.alpha * max(1, len(G_active_patterns)):
                    period = min(1, period / 2)
                else:
                    period += 1
                timer = 0

            G_next: Dict[int, MovingClusterPattern] = {}
            # 使用 MC2 的方式关联并扩展 G
            # object_index = {}
            # for cid, c in L.items():
            #     for oid in c.members:
            #         object_index[oid] = c
            # print('t: ', t, 'G_active_patterns', G_active_patterns)
            # 采用MC2算法中的快速相交判断方法扩展当前移动簇
            for mc_id, g in G_active_patterns.items():
                g.extended = False
                # if g.removed:
                #     continue
                # 获取该模式中上一个时刻的对象
                g_last_t_objects = g.time_to_objects[sorted_times[idx - 1]]
                k = (1 - self.theta) * len(g_last_t_objects)
                while k > 0:
                    o_j = random.choice(list(g_last_t_objects))
                    c_res = self.find_object_in_clusters(o_j, L)
                    if not c_res[0]:
                        k = k - 1
                    else:
                        c = c_res[1]
                        inter = len(g_last_t_objects & set(c.members))
                        union = len(g_last_t_objects | set(c.members))
                        if union > 0 and inter / union >= self.theta:
                            g.extended = True
                            g.extend_pattern(t, set(c.members))
                            G_next[mc_id] = g
                            c.assigned = True
                            break
                # 如果当前时刻簇和移动簇相交判断结束后,某个移动簇没有被当前时刻的簇扩展, 当前判断其是否是一个有效的移动簇, 如果是则输出
                if not g.extended:
                    if g.lifetime >= self.k:
                        G_final_patterns[mc_id] = g
                    else:
                        g.removed = True

            # 判断当前簇是否被已有移动簇, 如果没有则新建一个移动簇
            for cid, c in L.items():
                if not c.assigned:
                    # 新建一个moving cluster
                    mc = MovingClusterPattern()
                    mc.extend_pattern(t, set(c.members))
                    pattern_counter += 1
                    mc.pattern_counter = pattern_counter
                    G_next[pattern_counter] = mc

            G_active_patterns = G_next
            # print(G_active_patterns)
            # 收尾: 剩余活跃模式判断是否满足条件
        for mc_id, g in G_active_patterns.items():
            if g.lifetime > self.k:
                G_final_patterns[mc_id] = g

        return G_final_patterns

    def find_object_in_clusters(self, o: str, clusters: Dict[str, ClusterClass]):
        for cid, c in clusters.items():
            if o in c.members:
                return True, c
        return False, []

    def dbscan(self, points, t):
        data = np.array([[p.x, p.y] for p in points])
        labels = DBSCAN(eps=self.eps, min_samples=self.minPts, metric='euclidean').fit(data).labels_
        clusters: Dict[int, ClusterClass] = {}
        # clusters = []
        for i, label in enumerate(labels):
            if label != -1:
                if label not in clusters.keys():
                    cluster = ClusterClass(cid=label, t=t, members=[points[i].oid])
                    clusters[label] = cluster
                else:
                    clusters[label].add_member(points[i].oid)
        return clusters
