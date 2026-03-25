#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: trajectory_relation.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:26
# moveminerx/spatial_relation/trajectory_relation.py

import numpy as np
from shapely.geometry import LineString, Point
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial import distance


class TrajectorySpatialRelation:

    @staticmethod
    def euclidean(traj1, traj2):
        coords1 = [(p.x, p.y) for p in traj1.points]
        coords2 = [(p.x, p.y) for p in traj2.points]
        return np.linalg.norm(np.array(coords1) - np.array(coords2))

    @staticmethod
    def dtw(traj1, traj2):
        coords1 = [(p.x, p.y) for p in traj1.points]
        coords2 = [(p.x, p.y) for p in traj2.points]
        distance, path = fastdtw(coords1, coords2, dist=euclidean)
        return distance

    @staticmethod
    def lcss(traj1, traj2, epsilon=5.0):
        """
        LCSS距离
        """
        coords1 = [(p.x, p.y) for p in traj1.points]
        coords2 = [(p.x, p.y) for p in traj2.points]
        n = len(coords1)
        m = len(coords2)
        L = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if np.linalg.norm(np.array(coords1[i - 1]) - np.array(coords2[j - 1])) < epsilon:
                    L[i, j] = L[i - 1, j - 1] + 1
                else:
                    L[i, j] = max(L[i - 1, j], L[i, j - 1])
        return 1 - L[n, m] / min(n, m)

    @staticmethod
    def discrete_frechet(traj1, traj2):
        """
        Discrete Frechet distance
        """
        from scipy.spatial.distance import cdist
        coords1 = np.array([(p.x, p.y) for p in traj1.points])
        coords2 = np.array([(p.x, p.y) for p in traj2.points])
        C = cdist(coords1, coords2)
        ca = np.full(C.shape, -1.0)

        def _c(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
            elif i == 0 and j == 0:
                ca[i, j] = C[0, 0]
            elif i > 0 and j == 0:
                ca[i, j] = max(_c(i - 1, 0), C[i, 0])
            elif i == 0 and j > 0:
                ca[i, j] = max(_c(0, j - 1), C[0, j])
            else:
                ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), C[i, j])
            return ca[i, j]

        return _c(len(coords1) - 1, len(coords2) - 1)

    @staticmethod
    def hausdorff(traj1, traj2):
        coords1 = np.array([(p.x, p.y) for p in traj1.points])
        coords2 = np.array([(p.x, p.y) for p in traj2.points])
        from scipy.spatial.distance import directed_hausdorff
        return max(directed_hausdorff(coords1, coords2)[0],
                   directed_hausdorff(coords2, coords1)[0])

    @staticmethod
    def one_way_distance(traj1, traj2):
        coords1 = np.array([(p.x, p.y) for p in traj1.points])
        coords2 = np.array([(p.x, p.y) for p in traj2.points])
        from scipy.spatial.distance import cdist
        dist = cdist(coords1, coords2)
        return np.mean(np.min(dist, axis=1))

    @staticmethod
    def sspd(traj1, traj2):
        """
        Symmetrised Segmented Path Distance
        """
        coords1 = np.array([(p.x, p.y) for p in traj1.points])
        coords2 = np.array([(p.x, p.y) for p in traj2.points])
        d12 = np.mean(np.min(np.linalg.norm(coords1[:, None] - coords2[None, :], axis=2), axis=1))
        d21 = np.mean(np.min(np.linalg.norm(coords2[:, None] - coords1[None, :], axis=2), axis=1))
        return 0.5 * (d12 + d21)


    @staticmethod
    def sspd_star(traj1, traj2):
        """
        SSPD* 变体，带权重
        """
        coords1 = np.array([(p.x, p.y) for p in traj1.points])
        coords2 = np.array([(p.x, p.y) for p in traj2.points])
        d12 = np.mean(np.min(np.linalg.norm(coords1[:, None] - coords2[None, :], axis=2) ** 2, axis=1))
        d21 = np.mean(np.min(np.linalg.norm(coords2[:, None] - coords1[None, :], axis=2) ** 2, axis=1))
        return 0.5 * (d12 + d21)

    @staticmethod
    def dspd(traj1, traj2):
        """
        Directed Segmented Path Distance
        """
        coords1 = np.array([(p.x, p.y) for p in traj1.points])
        coords2 = np.array([(p.x, p.y) for p in traj2.points])
        d12 = np.mean(np.min(np.linalg.norm(coords1[:, None] - coords2[None, :], axis=2), axis=1))
        return d12
