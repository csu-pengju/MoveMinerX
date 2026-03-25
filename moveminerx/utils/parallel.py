#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: parallel.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 12:29
# moveminerx/utils/parallel.py

from multiprocessing import Pool, cpu_count
from functools import partial


def run_parallel(func, data_list, n_jobs=None):
    """
    并行执行函数
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    with Pool(n_jobs) as pool:
        results = pool.map(func, data_list)

    return results


def run_parallel_with_args(func, data_list, **kwargs):
    """
    支持额外参数的并行执行
    """
    func_with_args = partial(func, **kwargs)
    return run_parallel(func_with_args, data_list)