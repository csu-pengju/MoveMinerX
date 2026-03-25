#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: co_location_miner.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:48

from moveminerx.mining.base import BaseMiner

class CoLocationMiner(BaseMiner):
    """
    同位模式挖掘
    """
    def __init__(self, min_overlap=0.5):
        self.min_overlap = min_overlap
        self.patterns = []

    def fit(self, trajectories):
        # TODO: 基于时间空间重叠 / 相似轨迹子序列
        return self

    def get_patterns(self):
        return self.patterns