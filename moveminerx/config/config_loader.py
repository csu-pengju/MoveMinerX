#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: config_loader.py.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 11:38
# moveminerx/config/config_loader.py

import yaml
import os
from copy import deepcopy


class Config:
    """
    MoveMinerX 配置类（支持点访问）
    """

    def __init__(self, config_dict):
        self._config = config_dict

    def __getitem__(self, key):
        return self._config.get(key)

    def __getattr__(self, key):
        value = self._config.get(key)
        if isinstance(value, dict):
            return Config(value)
        return value

    def to_dict(self):
        return self._config

    def __repr__(self):
        return f"<Config {self._config}>"


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_dicts(base, override):
    """
    递归合并配置（override覆盖base）
    """
    result = deepcopy(base)

    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v

    return result


def load_config(config_path=None, default_path=None):
    """
    加载配置（支持默认 + 用户自定义）
    """

    # 默认配置路径
    if default_path is None:
        default_path = os.path.join(
            os.path.dirname(__file__), "default.yaml"
        )

    default_cfg = load_yaml(default_path)

    if config_path is None:
        return Config(default_cfg)

    user_cfg = load_yaml(config_path)

    merged_cfg = merge_dicts(default_cfg, user_cfg)
    from .schema import validate_config
    validate_config(merged_cfg)

    return Config(merged_cfg)


