#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: pipeline.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 11:54

import time
# 工作流调度, 支持: 串行处理, 日志输出, 可扩展（并行/分支）


class Pipeline:
    """
    MoveMinerX 核心流水线
    """

    def __init__(self, steps=None, verbose=True):
        self.steps = steps or []
        self.verbose = verbose

    def add(self, step):
        self.steps.append(step)

    def run(self, data):
        """
        顺序执行 pipeline
        """
        for step in self.steps:
            start = time.time()

            if self.verbose:
                print(f"[Pipeline] Running step: {step}")

            data = step.run(data)

            if self.verbose:
                print(f"[Pipeline] Done in {time.time() - start:.4f}s\n")

        return data

    def __repr__(self):
        step_names = [step.__class__.__name__ for step in self.steps]
        return f"<Pipeline steps={step_names}>"


