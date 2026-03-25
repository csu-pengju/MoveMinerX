#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: anomaly_miner.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:48
from moveminerx.mining.base import BaseMiner
from moveminerx.spatial_relation import TrajectorySpatialRelation, FlowSpatialRelation

class TrajectoryAnomalyMiner(BaseMiner):
    """
    异常轨迹检测
    """
    def __init__(self, threshold=100):
        self.threshold = threshold
        self.patterns = []

    def fit(self, trajectories):
        # 简单占位：轨迹长度异常
        for traj in trajectories:
            length = sum(np.linalg.norm(np.array([traj.points[i].x,traj.points[i].y]) -
                                        np.array([traj.points[i-1].x,traj.points[i-1].y]))
                        for i in range(1,len(traj.points)))
            if length > self.threshold:
                self.patterns.append(traj)
        return self

    def get_patterns(self):
        return self.patterns


class FlowAnomalyMiner(BaseMiner):
    """
    异常流检测
    """
    def __init__(self, threshold=50):
        self.threshold = threshold
        self.patterns = []

    def fit(self, flows):
        # TODO: 基于流量异常 / 方向异常
        return self

    def get_patterns(self):
        return self.patterns