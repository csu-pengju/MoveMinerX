#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: base.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 14:52

from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """
    所有预处理器的基类
    """

    @abstractmethod
    def clean(self, data):
        """
        清理噪声数据
        """
        pass

    @abstractmethod
    def preprocess(self, data):
        """
        对数据进行完整预处理
        """
        pass
