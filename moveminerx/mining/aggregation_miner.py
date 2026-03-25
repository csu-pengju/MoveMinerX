#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: aggregation_miner.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:48

from moveminerx.mining.base import BaseMiner
from sklearn.cluster import DBSCAN, KMeans


class PointClusterMiner(BaseMiner):
    """
    点聚集模式挖掘
    """
    def __init__(self, eps=50, min_samples=3):
        self.eps = eps
        self.min_samples = min_samples
        self.patterns = []

    def fit(self, points):
        coords = [(p.x, p.y) for p in points]
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(coords)
        labels = clustering.labels_
        # 组织聚类结果
        clusters = {}
        for label, point in zip(labels, points):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(point)
        self.patterns = list(clusters.values())
        return self

    def get_patterns(self):
        return self.patterns


class LineClusterMiner(BaseMiner):
    """
    线聚集模式（Flow / Trajectory segments）挖掘
    """
    def __init__(self):
        self.patterns = []

    def fit(self, lines):
        # TODO: 可基于相似度矩阵 + 层次聚类 / DBSCAN
        return self

    def get_patterns(self):
        return self.patterns