#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: utils.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:10

import numpy as np


def douglas_peucker(coords, epsilon):
    """
    Douglas-Peucker算法简化线
    coords: [(x, y), ...]
    """
    if len(coords) < 3:
        return coords

    # 递归实现
    start, end = coords[0], coords[-1]
    dmax = 0.0
    index = 0
    for i in range(1, len(coords) - 1):
        d = perpendicular_distance(coords[i], start, end)
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        # 递归
        rec1 = douglas_peucker(coords[:index+1], epsilon)
        rec2 = douglas_peucker(coords[index:], epsilon)
        return rec1[:-1] + rec2
    else:
        return [start, end]

def perpendicular_distance(point, start, end):
    """
    计算点到线段距离
    """
    x0, y0 = point
    x1, y1 = start
    x2, y2 = end
    num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    den = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
    return num / den if den != 0 else 0

def linear_interpolation(traj, method="linear", interval=1.0):
    """
    线性插值（简化版）
    """
    points = traj.points
    new_points = []
    for i in range(len(points)-1):
        p1, p2 = points[i], points[i+1]
        dt = getattr(p2.t, "total_seconds", lambda :1)() - getattr(p1.t, "total_seconds", lambda :0)()
        steps = max(int(dt / interval), 1)
        for s in range(steps):
            ratio = s / steps
            x = p1.x + (p2.x - p1.x) * ratio
            y = p1.y + (p2.y - p1.y) * ratio
            t = p1.t + (p2.t - p1.t) * ratio
            new_points.append(type(p1)(p1.obj_id, x, y, t))
    new_points.append(points[-1])
    traj.points = new_points
    return traj