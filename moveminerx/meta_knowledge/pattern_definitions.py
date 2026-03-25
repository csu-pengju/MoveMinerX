#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: pattern_definitions.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 16:25
# moveminerx/meta_knowledge/pattern_definitions.py

from moveminerx.meta_knowledge.base import BasePatternMeta
import numpy as np
from moveminerx.spatial_relation import TrajectorySpatialRelation, FlowSpatialRelation, PointSpatialRelation

class ConvoyMeta(BasePatternMeta):
    """
    伴随模式（Convoy）
    """
    def __init__(self, min_members=3, min_length=3):
        description = "一组对象在至少 min_length 个连续时间戳内，始终以一定距离伴随"
        constraints = {
            'min_members': min_members,
            'min_length': min_length,
            'spatial': '距离阈值内伴随',
            'temporal': '连续时间戳',
            'motion': '轨迹方向/速度相似',
            'membership': '成员固定或动态变化',
        }
        attributes = {'avg_member_count': None, 'avg_duration': None, 'avg_spatial_density': None}
        super().__init__("Convoy", description, constraints, attributes)

    def validate(self, pattern):
        """
        验证模式是否满足元知识约束
        """
        if len(pattern) < self.constraints['min_members']:
            return False
        durations = [len(traj.points) for traj in pattern]
        if min(durations) < self.constraints['min_length']:
            return False
        # TODO: 可扩展加入空间紧密度/方向一致性验证
        return True


class FlockMeta(BasePatternMeta):
    """
    Flock 模式
    """
    def __init__(self, min_members=3, radius=50):
        description = "一组对象在一定半径范围内伴随移动至少一定时间"
        constraints = {
            'min_members': min_members,
            'radius': radius,
            'temporal': '连续时间戳',
            'spatial': '圆形范围内',
        }
        super().__init__("Flock", description, constraints)

    def validate(self, pattern):
        # TODO: 可加入半径检测
        if len(pattern) < self.constraints['min_members']:
            return False
        return True


class PointClusterMeta(BasePatternMeta):
    """
    点聚集模式
    """
    def __init__(self, min_samples=3):
        description = "空间上紧密聚集的一组点"
        constraints = {
            'min_samples': min_samples,
            'spatial': '簇内点间距离小',
        }
        super().__init__("PointCluster", description, constraints)

    def validate(self, cluster):
        return len(cluster) >= self.constraints['min_samples']