#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: dataset.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 13:53


class Dataset:
    """
    统一数据集对象
    """

    def __init__(self, data, data_type="trajectory", name=None):
        self.data = data
        self.data_type = data_type
        self.name = name
        self.spatial_index = None

    def __len__(self):
        return len(self.data)

    def summary(self):
        return {
            "name": self.name,
            "type": self.data_type,
            "size": len(self.data)
        }

    def build_spatial_index(self):
        """
        为数据构建 R-tree 索引
        """
        from moveminerx.data_management.spatial_index.rtree_index import RTreeIndex
        from shapely.geometry import Point, LineString

        self.spatial_index = RTreeIndex()
        if self.data_type == "point":
            for p in self.data:
                self.spatial_index.insert(p.obj_id, Point(p.x, p.y))
        elif self.data_type == "trajectory":
            for traj in self.data:
                line = LineString([(p.x, p.y) for p in traj.points])
                self.spatial_index.insert(traj.obj_id, line)
        elif self.data_type == "flow":
            for flow in self.data:
                self.spatial_index.insert(flow.flow_id, flow.geometry)
        else:
            raise ValueError(f"Unsupported data_type {self.data_type}")

    def compute_spatial_relation(self, relation_type, radius=None, **kwargs):
        """
        统一空间关系计算接口
        - relation_type: string, 例如 "euclidean", "dtw", "average_distance"
        - radius: optional, 半径邻近加速
        - kwargs: 传给具体算法参数
        """
        from moveminerx.spatial_relation import (
            PointSpatialRelation, TrajectorySpatialRelation, FlowSpatialRelation
        )
        results = dict()

        if self.spatial_index is None and radius is not None:
            self.build_spatial_index()

        N = len(self.data)
        for i in range(N):
            results[i] = dict()
            # 如果启用半径邻近搜索
            neighbors = range(N)
            if radius is not None:
                geom_i = self.spatial_index.data[
                    self.data[i].obj_id if self.data_type == "point" else getattr(self.data[i], "traj_id",
                                                                                  self.data[i].flow_id)]
                neighbor_ids = self.spatial_index.query_radius((geom_i.centroid.x, geom_i.centroid.y), radius)
                neighbors = neighbor_ids

            for j in neighbors:
                if i == j:
                    continue
                if self.data_type == "point":
                    func = getattr(PointSpatialRelation, relation_type)
                    results[i][j] = func(self.data[i], self.data[j])
                elif self.data_type == "trajectory":
                    func = getattr(TrajectorySpatialRelation, relation_type)
                    results[i][j] = func(self.data[i], self.data[j], **kwargs)
                elif self.data_type == "flow":
                    func = getattr(FlowSpatialRelation, relation_type)
                    results[i][j] = func(self.data[i], self.data[j], **kwargs)

        return results
