#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: timer.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 12:28
# moveminerx/utils/timer.py

import time
from contextlib import contextmanager


@contextmanager
def timer(name="Block"):
    start = time.time()
    yield
    end = time.time()
    print(f"[Timer] {name}: {end - start:.4f}s")