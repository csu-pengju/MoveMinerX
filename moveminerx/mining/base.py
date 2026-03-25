#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: base.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:47

from abc import ABC, abstractmethod


class BaseMiner(ABC):
    """
    所有运动模式挖掘算法的基类
    """

    @abstractmethod
    def fit(self, data):
        """
        训练 / 挖掘模式
        """
        pass

    @abstractmethod
    def get_patterns(self):
        """
        返回挖掘到的模式
        """
        pass
