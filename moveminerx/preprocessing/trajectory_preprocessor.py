#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: trajectory_preprocessor.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 14:53
# moveminerx/preprocessing/trajectory_preprocessor.py

import numpy as np
from shapely.geometry import LineString
from moveminerx.preprocessing.base import BasePreprocessor
from moveminerx.preprocessing.utils import douglas_peucker, linear_interpolation
from moveminerx.utils import timer


class TrajectoryPreprocessor(BasePreprocessor):
    """
    Trajectory 预处理，包括：
    - Trajectory cleaning
    - Interpolation
    - Map matching
    - Simplification
    - Time alignment
    """

    def __init__(self, min_points=3, min_length=10):
        self.min_points = min_points
        self.min_length = min_length

    def clean(self, trajectories):
        """
        清理异常轨迹或点
        """
        clean_trajs = []
        for traj in trajectories:
            # 1️⃣ 移除异常点
            points = [p for p in traj.points if p is not None]

            # 2️⃣ 移除异常轨迹
            line = LineString([(p.x, p.y) for p in points])
            if len(points) < self.min_points or line.length < self.min_length:
                continue
            traj.points = points
            clean_trajs.append(traj)
        return clean_trajs

    def interpolate(self, traj, method="linear", interval=1.0):
        """
        插值
        method: linear / spline / lagrange
        interval: 插值时间间隔
        """
        return linear_interpolation(traj, method, interval)

    def simplify(self, traj, epsilon=5.0):
        """
        Douglas-Peucker 简化
        """
        coords = [(p.x, p.y) for p in traj.points]
        simplified = douglas_peucker(coords, epsilon)
        traj.points = [traj.points[i] for i, _ in enumerate(simplified)]
        return traj

    def map_match(self, traj, map_data=None):
        """
        HMM map matching placeholder
        """
        # TODO: 可以调用外部HMM map matching库
        return traj

    def time_align(self, traj):
        """
        时间对齐（统一时间戳）
        """
        return traj

    def preprocess(self, trajectories):
        processed = []
        for traj in trajectories:
            traj = self.clean([traj])[0]
            traj = self.interpolate(traj)
            traj = self.simplify(traj)
            traj = self.map_match(traj)
            traj = self.time_align(traj)
            processed.append(traj)
        return processed