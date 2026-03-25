#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: registry.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 11:54
from typing import Dict, Type
# 插件注册机制支持：动态注册算法与用户扩展（开源关键）


class Registry:
    """
    通用注册器（支持 miner / metric / preprocessor 等）
    """

    def __init__(self, name):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, key: str):
        def decorator(cls):
            if key in self._registry:
                raise KeyError(f"{key} already registered in {self.name}")
            self._registry[key] = cls
            return cls
        return decorator

    def get(self, key: str):
        if key not in self._registry:
            raise KeyError(f"{key} not found in {self.name}")
        return self._registry[key]

    def list(self):
        return list(self._registry.keys())

    def __repr__(self):
        return f"<Registry {self.name}: {list(self._registry.keys())}>"


# 创建全局注册器
MINER_REGISTRY = Registry("miner")
METRIC_REGISTRY = Registry("metric")
PREPROCESSOR_REGISTRY = Registry("preprocessor")

