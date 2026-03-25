#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: trajectory.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 12:40
from typing import List
from .base import BaseData
from .point import MovingPoint


class Trajectory(BaseData):
    def __init__(self, obj_id, points: List[MovingPoint]):
        super().__init__(obj_id)
        self.points = points  # list of MovingPoint

    def __len__(self):
        return len(self.points)

    def get_coords(self):
        return [(p.x, p.y) for p in self.points]

    def get_times(self):
        return [p.t for p in self.points]
