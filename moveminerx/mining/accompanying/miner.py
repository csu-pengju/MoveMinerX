#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: miner.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 17:41
# moveminerx/mining/convoy/miner.py

from moveminerx.mining.base import BaseMiner
from ._convoy import CMCAlgorithm, CuTSAlgorithm, CuTSStarAlgorithm, CuTSPlusAlgorithm
from ._flock import MovingFlockMiner
from ._moving_cluster import MCAlgorithm
from ._swarm import SwarmMiner


class AccompanyingMiner(BaseMiner):
    """
    Convoy 统一调用接口
    """

    def __init__(self, method="cmc", pattern='convoy', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.patterns = []

    def fit(self, trajectories):
        if self.method == "cmc":
            algo = CMCAlgorithm(**self.kwargs)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        self.patterns = algo.run()
        return self

    def get_patterns(self):
        return self.patterns
