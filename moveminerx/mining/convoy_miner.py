#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: convoy_miner.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:48

from moveminerx.mining.base import BaseMiner
from moveminerx.spatial_relation import TrajectorySpatialRelation


class ConvoyMiner(BaseMiner):
    """
    Convoy 模式挖掘
    """
    def __init__(self, min_members=3, min_length=3):
        self.min_members = min_members
        self.min_length = min_length
        self.patterns = []

    def fit(self, trajectories):
        # TODO: 基于 DBSCAN / Incremental Convoy 算法
        # 占位示例
        for i, traj in enumerate(trajectories):
            # 简单占位逻辑
            if len(traj.points) >= self.min_length:
                self.patterns.append([traj])
        return self

    def get_patterns(self):
        return self.patterns


class FlockMiner(BaseMiner):
    """
    Flock 模式挖掘
    """
    def __init__(self, min_members=3, radius=50):
        self.min_members = min_members
        self.radius = radius
        self.patterns = []

    def fit(self, trajectories):
        # TODO: 基于圆形区域 Flock 挖掘
        return self

    def get_patterns(self):
        return self.patterns
