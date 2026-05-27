#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: definition.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 17:40

class ConvoyPattern:
    """
    Convoy 模式定义
    """

    def __init__(self, members, start_time=None, end_time=None):
        self.members = members
        self.start_time = start_time
        self.end_time = end_time

    def duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def __repr__(self):
        return f"<Convoy size={len(self.members)}>"

