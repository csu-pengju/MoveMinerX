#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: point_relation.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:26
# moveminerx/spatial_relation/point_relation.py

import numpy as np
from geopy.distance import geodesic
from shapely.geometry import Point


class PointSpatialRelation:

    @staticmethod
    def euclidean(p1, p2):
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return np.sqrt(dx * dx + dy * dy)

    @staticmethod
    def manhattan(p1, p2):
        dx = abs(p1.x - p2.x)
        dy = abs(p1.y - p2.y)
        return dx + dy

    @staticmethod
    def great_circle(p1, p2):
        return geodesic((p1.y, p1.x), (p2.y, p2.x)).meters

    @staticmethod
    def cosine_similarity(p1, p2):
        v1 = np.array([p1.x, p1.y])
        v2 = np.array([p2.x, p2.y])
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    @staticmethod
    def network_distance(p1, p2, network):
        """
        计算网络距离
        network: networkx Graph
        p1, p2: MovingPoint
        """
        import networkx as nx
        nearest1 = network.graph['nodes_nearest'][p1.obj_id]
        nearest2 = network.graph['nodes_nearest'][p2.obj_id]
        return nx.shortest_path_length(network, nearest1, nearest2, weight='length')
