#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: geometry_utils.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/10/21 16:34
import math
from typing import Tuple, Optional


def normalize(vx: float, vy: float) -> Tuple[float, float]:
    norm = math.hypot(vx, vy)
    if norm == 0:
        return 0.0, 0.0
    return vx / norm, vy / norm


def line_from_point_dir(x0: float, y0: float, dx: float, dy: float):
    """返回直线参数 (A,B,C) 表示 Ax + By + C = 0"""
    # 方向向量 (dx,dy) -> 法向量 (-dy, dx)
    A = -dy
    B = dx
    C = -(A * x0 + B * y0)
    return A, B, C


def intersect_lines(x1, y1, dx1, dy1, x2, y2, dx2, dy2) -> Optional[Tuple[float, float]]:
    """
    计算通过 (x1,y1) 方向 (dx1,dy1) 的直线与第二条直线的交点（无穷延拓直线）。
    如果平行返回 None。
    """
    A1, B1, C1 = line_from_point_dir(x1, y1, dx1, dy1)
    A2, B2, C2 = line_from_point_dir(x2, y2, dx2, dy2)
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-12:
        return None
    x = (B1 * C2 - B2 * C1) / det
    y = (C1 * A2 - C2 * A1) / det
    return (x, y)


def point_is_in_front_of_ray(px, py, rx, ry, dx, dy) -> bool:
    """判断点 p=(px,py) 是否位于从 r=(rx,ry) 沿方向 (dx,dy) 的半直线前方"""
    vx = px - rx
    vy = py - ry
    dot = vx * dx + vy * dy
    return dot >= -1e-9  # 允许微小数值误差


def dist_point_to_infinite_line(px, py, x0, y0, dx, dy) -> float:
    """点到无限直线的垂直距离（方向向量不必单位化）"""
    A, B, C = line_from_point_dir(x0, y0, dx, dy)
    return abs(A * px + B * py + C) / math.hypot(A, B)
