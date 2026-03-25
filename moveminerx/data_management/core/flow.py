#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: flow.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 12:43

from shapely.geometry import LineString


class Flow:
    """
    Flow 表示为一条线（OD flow or trajectory segment）
    """

    def __init__(self, coords, flow_id=None, volume=1, attributes=None):
        """
        coords: [(x1, y1), (x2, y2), ...]
        """
        self.flow_id = flow_id
        self.geometry = LineString(coords)
        self.origin = coords[0]
        self.destination = coords[-1]
        self.volume = volume
        self.attributes = attributes or {}

    def length(self):
        return self.geometry.length

    def to_wkt(self):
        return self.geometry.wkt

    def __repr__(self):
        return f"<Flow id={self.flow_id}, length={self.length():.2f}>"

