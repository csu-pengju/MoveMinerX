#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: base.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:25

from abc import ABC, abstractmethod

class BaseSpatialRelation(ABC):
    """
    空间关系计算基类
    """

    @abstractmethod
    def compute(self, obj1, obj2):
        """
        计算两个对象的空间关系
        """
        pass