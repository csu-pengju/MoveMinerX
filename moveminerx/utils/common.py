#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: common.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 12:29
# moveminerx/utils/common.py

import random
import numpy as np


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def ensure_list(x):
    if isinstance(x, list):
        return x
    return [x]


def dict_to_str(d):
    return "_".join(f"{k}={v}" for k, v in d.items())