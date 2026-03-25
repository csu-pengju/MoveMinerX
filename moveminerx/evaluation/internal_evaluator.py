#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: internal_evaluator.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:54
# moveminerx/evaluation/internal_evaluator.py

import numpy as np
from moveminerx.spatial_relation import TrajectorySpatialRelation, FlowSpatialRelation, PointSpatialRelation
from moveminerx.evaluation.base import BaseEvaluator


class InternalEvaluator(BaseEvaluator):
    """
    内部指标评估
    """
    def __init__(self, mode_type='trajectory'):
        self.mode_type = mode_type

    def evaluate(self, patterns, ground_truth=None):
        """
        针对不同模式类型计算内部指标
        """
        if self.mode_type in ['convoy', 'flock', 'swarm']:
            return self._evaluate_convoy(patterns)
        elif self.mode_type in ['point_cluster', 'line_cluster', 'trajectory_cluster', 'flow_cluster']:
            return self._evaluate_cluster(patterns)
        elif self.mode_type in ['anomaly']:
            return self._evaluate_anomaly(patterns)
        else:
            raise ValueError(f"Unsupported mode_type {self.mode_type}")

    def _evaluate_convoy(self, patterns):
        """
        内部指标：
        - 成员持续时间（平均轨迹长度）
        - 成员一致性（平均每个模式成员数）
        - 空间紧密度（平均轨迹间距离）
        """
        durations = []
        member_counts = []
        densities = []

        for pattern in patterns:
            member_counts.append(len(pattern))
            traj_lengths = [len(traj.points) for traj in pattern]
            durations.append(np.mean(traj_lengths))

            # 空间紧密度
            dist_list = []
            for i in range(len(pattern)):
                for j in range(i+1, len(pattern)):
                    dist_list.append(TrajectorySpatialRelation.one_way_distance(pattern[i], pattern[j]))
            densities.append(np.mean(dist_list) if dist_list else 0)

        metrics = {
            'avg_duration': np.mean(durations) if durations else 0,
            'avg_member_count': np.mean(member_counts) if member_counts else 0,
            'avg_spatial_density': np.mean(densities) if densities else 0
        }
        return metrics

    def _evaluate_cluster(self, patterns):
        """
        聚集模式内部指标：
        - 簇内紧密度：簇内对象平均距离
        - 簇间分离度：簇心间距离
        """
        intra_distances = []
        cluster_centers = []

        for cluster in patterns:
            coords = [(p.x, p.y) for p in cluster]
            if len(coords) <= 1:
                intra_distances.append(0)
                cluster_centers.append(np.array(coords[0]))
                continue
            dist_matrix = np.linalg.norm(np.array(coords)[:, None, :] - np.array(coords)[None, :, :], axis=2)
            intra_distances.append(np.mean(dist_matrix))
            cluster_centers.append(np.mean(coords, axis=0))

        # 簇间分离度
        inter_distances = []
        for i in range(len(cluster_centers)):
            for j in range(i+1, len(cluster_centers)):
                inter_distances.append(np.linalg.norm(cluster_centers[i]-cluster_centers[j]))

        metrics = {
            'avg_intra_distance': np.mean(intra_distances) if intra_distances else 0,
            'avg_inter_distance': np.mean(inter_distances) if inter_distances else 0
        }
        return metrics

    def _evaluate_anomaly(self, patterns):
        """
        异常模式内部指标：
        - 异常程度均值
        - 异常轨迹比例
        """
        scores = [getattr(p, 'anomaly_score', 1) for p in patterns]  # 假设对象有 anomaly_score 属性
        metrics = {
            'avg_anomaly_score': np.mean(scores) if scores else 0,
            'anomaly_count': len(patterns)
        }
        return metrics