#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: base.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:54
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """
    所有模式评估器的基类
    """
    @abstractmethod
    def evaluate(self, patterns, ground_truth=None):
        """
        评估挖掘结果
        patterns: 挖掘得到的模式列表
        ground_truth: 真值模式（可选）
        返回评价指标字典
        """
        pass
