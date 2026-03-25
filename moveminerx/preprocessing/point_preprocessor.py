#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: point_preprocessor.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 14:52
import numpy as np
from moveminerx.preprocessing.base import BasePreprocessor
from shapely.geometry import Point
from moveminerx.utils import logger, timer


class PointPreprocessor(BasePreprocessor):
    """
    MovingPoint 预处理，包括：
    - 噪声过滤
    - 离群点移除
    """

    def __init__(self, distance_threshold=100, speed_threshold=50):
        """
        distance_threshold: 单位m，异常点跳变阈值
        speed_threshold: 单位m/s，异常速度阈值
        """
        self.distance_threshold = distance_threshold
        self.speed_threshold = speed_threshold

    def clean(self, points):
        """
        清理噪声点和离群点
        """
        clean_points = []
        for i, p in enumerate(points):
            if i == 0:
                clean_points.append(p)
                continue

            prev = clean_points[-1]
            dx = p.x - prev.x
            dy = p.y - prev.y
            dt = (p.t - prev.t).total_seconds() if hasattr(p.t, 'total_seconds') else 1
            dist = np.sqrt(dx**2 + dy**2)
            speed = dist / dt if dt > 0 else 0

            if dist > self.distance_threshold or speed > self.speed_threshold:
                continue
            clean_points.append(p)

        return clean_points

    def preprocess(self, points):
        # 可以扩展更多操作
        return self.clean(points)