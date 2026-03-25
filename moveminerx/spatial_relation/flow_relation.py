#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: flow_relation.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:26
# moveminerx/spatial_relation/flow_relation.py

import numpy as np
from shapely.geometry import LineString, Point


class FlowSpatialRelation:

    @staticmethod
    def maximum_distance(f1, f2):
        return max(f1.geometry.distance(Point(c)) for c in f2.geometry.coords)

    @staticmethod
    def summed_distance(f1, f2):
        return sum(f1.geometry.distance(Point(c)) for c in f2.geometry.coords)

    @staticmethod
    def average_distance(f1, f2):
        return np.mean([f1.geometry.distance(Point(c)) for c in f2.geometry.coords])

    @staticmethod
    def weighted_distance(f1, f2, weights=None):
        coords = f2.geometry.coords
        if weights is None:
            weights = np.ones(len(coords))
        distances = [f1.geometry.distance(Point(c)) for c in coords]
        return np.average(distances, weights=weights)