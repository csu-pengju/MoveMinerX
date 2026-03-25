#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: convergence_miner.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:48

from moveminerx.mining.base import BaseMiner


class ConvergenceMiner(BaseMiner):
    """
    汇聚模式挖掘
    """
    def __init__(self, eps=50, min_members=3):
        self.eps = eps
        self.min_members = min_members
        self.patterns = []

    def fit(self, trajectories):
        # TODO: 基于轨迹终点聚集 / 相似方向
        return self

    def get_patterns(self):
        return self.patterns
