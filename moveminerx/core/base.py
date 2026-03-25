#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: base.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 11:54

from abc import ABC, abstractmethod


# 抽象基类
class BaseComponent(ABC):
    """
    所有模块的统一父类（Pipeline中的step）
    """

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def run(self, data):
        """
        执行模块逻辑
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


class BaseMiner(ABC):
    """
    所有运动模式挖掘算法的统一接口
    """

    def __init__(self, **params):
        self.params = params
        self.patterns = []
        self.is_fitted = False

    @abstractmethod
    def fit(self, dataset):
        """
        训练/挖掘过程
        """
        pass

    def get_patterns(self):
        if not self.is_fitted:
            raise RuntimeError("Miner has not been fitted yet.")
        return self.patterns

    def fit_predict(self, dataset):
        self.fit(dataset)
        return self.get_patterns()

    def __repr__(self):
        return f"<{self.__class__.__name__}(params={self.params})>"


class BaseMetric(ABC):
    """
    评估指标基类
    """

    @abstractmethod
    def compute(self, patterns):
        pass


class BasePreprocessor(BaseComponent):
    """
    预处理模块基类
    """
    pass



