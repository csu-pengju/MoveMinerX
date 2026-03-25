#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: base.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 16:24
from abc import ABC, abstractmethod

class BasePatternMeta(ABC):
    """
    所有运动模式元数据基类
    """

    def __init__(self, name, description, constraints=None, attributes=None):
        self.name = name                    # 模式名称
        self.description = description      # 模式定义描述
        self.constraints = constraints or {}  # 空间、时间、运动属性、对象成员、环境约束
        self.attributes = attributes or {}  # 模式特征，例如轨迹长度、成员数、密度等

    @abstractmethod
    def validate(self, pattern):
        """
        验证一个挖掘出的模式是否符合该模式元知识约束
        """
        pass