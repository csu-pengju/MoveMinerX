#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: base_loader.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 13:38

from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """
    所有数据加载器的统一接口
    """

    @abstractmethod
    def load_points(self, path, **kwargs):
        pass

    @abstractmethod
    def load_trajectories(self, path, **kwargs):
        pass

    @abstractmethod
    def load_flows(self, path, **kwargs):
        pass