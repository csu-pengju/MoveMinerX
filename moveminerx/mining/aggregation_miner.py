#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: aggregation_miner.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:48
import numpy as np

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

    def fit(self, points, method='DBSCAN', metric='precomputed', **kwargs):
        if metric == 'precomputed':
            coords = []
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

    def fit(self, lines, method='DBSCAN', eps=50, min_samples=2, metric='euclidean', **kwargs):
        """Fit clustering model to a list of line segments."""
        method = method.lower()
        line_segments = [self._normalize_line(line) for line in lines]

        if method == 'dbscan':
            if metric == 'precomputed':
                dist_matrix = self._segment_distance_matrix(line_segments)
                clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dist_matrix)
            else:
                features = np.vstack([self._segment_feature(seg) for seg in line_segments])
                clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(features)
        elif method == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 3)
            features = np.vstack([self._segment_feature(seg) for seg in line_segments])
            clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(features)
        else:
            raise ValueError(f'Unsupported clustering method: {method}')

        labels = clustering.labels_
        clusters = {}
        for label, line in zip(labels, line_segments):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(line)

        self.patterns = list(clusters.values())
        return self

    def _normalize_line(self, line):
        if isinstance(line, LineSegment):
            return line
        if isinstance(line, (list, tuple)) and len(line) == 2:
            start, end = line
            if hasattr(start, 'x') and hasattr(start, 'y') and hasattr(end, 'x') and hasattr(end, 'y'):
                return LineSegment(start_point=start, end_point=end, actual_tolerance=0.0)
            if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)):
                p1 = TrajectoryPoint(start[0], start[1], time=0)
                p2 = TrajectoryPoint(end[0], end[1], time=0)
                return LineSegment(start_point=p1, end_point=p2, actual_tolerance=0.0)
        raise ValueError('Line must be a LineSegment or a pair of coordinate tuples')

    def _segment_feature(self, segment):
        p1 = segment.start_point
        p2 = segment.end_point
        dx = float(p2.x) - float(p1.x)
        dy = float(p2.y) - float(p1.y)
        length = np.hypot(dx, dy)
        if length < 1e-9:
            return np.array([float(p1.x), float(p1.y), 0.0, 0.0])
        return np.array([
            (float(p1.x) + float(p2.x)) / 2.0,
            (float(p1.y) + float(p2.y)) / 2.0,
            dx / length,
            dy / length
        ])

    def _segment_distance_matrix(self, segments):
        n = len(segments)
        matrix = np.zeros((n, n), dtype=float)
        line_geoms = [LineString([(seg.start_point.x, seg.start_point.y),
                                  (seg.end_point.x, seg.end_point.y)]) for seg in segments]
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(line_geoms[i].distance(line_geoms[j]))
                matrix[i, j] = dist
                matrix[j, i] = dist
        return matrix

    def get_patterns(self):
        return self.patterns