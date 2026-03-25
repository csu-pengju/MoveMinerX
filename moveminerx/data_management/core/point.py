#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: point.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 12:40

from .base import BaseData
from shapely.geometry import Point


class MovingPoint(BaseData):
    def __init__(self, obj_id, x, y, t):
        super().__init__(obj_id)
        self.x = x
        self.y = y
        self.t = t

    def to_tuple(self):
        return (self.x, self.y, self.t)
