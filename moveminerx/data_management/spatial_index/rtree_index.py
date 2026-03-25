#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: rtree_index.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 14:42
from rtree import index
from shapely.geometry import Point, LineString


class RTreeIndex:
    """
    空间索引封装
    支持 MovingPoint, Trajectory, Flow
    """

    def __init__(self):
        # 创建 R-tree
        self.idx = index.Index()
        self.data = {}  # id -> object

    def insert(self, obj_id, geometry):
        """
        插入对象
        geometry: shapely Point 或 LineString
        """
        bbox = geometry.bounds  # (minx, miny, maxx, maxy)
        self.idx.insert(obj_id, bbox)
        self.data[obj_id] = geometry

    def query_radius(self, point, radius):
        """
        查询点半径内所有对象
        point: (x, y)
        radius: float
        """
        query_geom = Point(point).buffer(radius)
        bbox = query_geom.bounds
        candidate_ids = list(self.idx.intersection(bbox))
        results = []
        for obj_id in candidate_ids:
            geom = self.data[obj_id]
            if geom.distance(Point(point)) <= radius:
                results.append(obj_id)
        return results

    def query_bbox(self, bbox):
        """
        查询 bbox 内所有对象
        bbox: (minx, miny, maxx, maxy)
        """
        return list(self.idx.intersection(bbox))

    def nearest(self, point, num=1):
        """
        最近邻查询
        """
        nearest_ids = list(self.idx.nearest(point + point, num))  # trick: point = (x,y) duplicated
        return nearest_ids
