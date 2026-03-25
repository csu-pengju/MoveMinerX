#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: registry.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 13:53
DATASET_REGISTRY = {}


def register_dataset(name):
    def decorator(func):
        DATASET_REGISTRY[name] = func
        return func
    return decorator


def load_dataset(name, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {name} not found")

    return DATASET_REGISTRY[name](**kwargs)