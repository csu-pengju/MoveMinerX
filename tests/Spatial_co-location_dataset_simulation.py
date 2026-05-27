#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: Spatiotemporal_co-location_dataset_simulation.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/4/20 10:25
from typing import Dict, Optional

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

ST_X = (0, 100)
ST_Y = (0, 100)

n_parent = 10
mean_n_child = 3

r_child = 2  # 子类点空间半径
t_child = 2  # 子类点时间半径

n_grandchild = 3  # A, B, C三种要素类型
r_grandchild = 1  # 孙子类点空间半径（同现关系半径）

noise_rate = 0.4

missing_rates = {
    'type_0': 0.2,  # A要素缺失率20%
    'type_1': 0.4,  # B要素缺失率20%
    'type_2': 0.3  # C要素缺失率50%
}

patterns = [
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'C'),
    ('A', 'B', 'C')
]

all_types = ['A', 'B', 'C', 'D', 'E']


def generate_parent_points(n):
    """
    # Step 1: 生成父类点 (完成随机过程)
    :param n: 父类点数量
    :return:
    """
    parents = []
    for _ in range(n):
        x = np.random.uniform(*ST_X)
        y = np.random.uniform(*ST_Y)
        parents.append((x, y))
    return parents


def generate_child_points(parent_points, mean_n_child):
    """
    Step 2: 为每个父类点生成子类点
    :param parent_points:
    :param mean_n_child: 每个父类点子类点数量的泊松分布均值
    :return: 所有子类点坐标
    """
    child_points = []
    for parent in parent_points:
        # 确定子类点数量
        n_child = np.random.poisson(mean_n_child)
        if n_child == 0:
            continue
        # 在圆柱体内随机生成子类点
        for _ in range(n_child):
            # 空间位置： 在圆盘内均匀分布
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.sqrt(np.random.uniform(0, 1)) * r_child  # 面积均匀分布
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)

            x = parent[0] + dx
            y = parent[1] + dy
            # 边界裁剪
            x = np.clip(x, ST_X[0], ST_X[1])
            y = np.clip(y, ST_Y[0], ST_Y[1])
            child_points.append((x, y))

    return child_points


def generate_grandchild_points_with_labels(child_points, n_grandchild, r_grandchild):
    """
    # Step 3: 为每个子类点生成孙子类点（不同时空要素类型）
    :param child_points: 子类点坐标
    :param n_grandchild: 每个子类点生成的时空要素类型数量（孙子类点数量）
    :param r_grandchild: 孙子类点空间半径
    :return: 各要素类型的点集字典
    """
    grandchild_dict = {f'type_{i}': [] for i in range(n_grandchild)}
    pattern_labels = {f'type_{i}': [] for i in range(n_grandchild)}

    for child_idx, child in enumerate(child_points):
        for type_idx in range(n_grandchild):
            # 在圆柱体内随机生成孙子类点
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.sqrt(np.random.uniform(0, 1)) * r_grandchild
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)

            x = child[0] + dx
            y = child[1] + dy

            # 边界裁剪
            x = np.clip(x, ST_X[0], ST_X[1])
            y = np.clip(y, ST_Y[0], ST_Y[1])
            grandchild_dict[f'type_{type_idx}'].append((x, y))

            # 记录模式标签
            # 由于同一父类点生成的所有孙子类点天然形成同现模式
            # 根据参与的类型组合打标签
            if n_grandchild == 3:  # A, B, C
                if type_idx == 0:  # A
                    label = f'child_{child_idx}_pattern'
                elif type_idx == 1:  # B
                    label = f'child_{child_idx}_pattern'
                else:  # C
                    label = f'child_{child_idx}_pattern'
                pattern_labels[f'type_{type_idx}'].append(label)

    return grandchild_dict, pattern_labels


def remove_missing_points(point_dict: Dict, pattern_labels, missing_rate):
    """
    Step 4: 移除部分点（模拟数据缺失）
    :param point_dict: 各要素类型的点集字典
    :param pattern_labels:  各要素类型的标签列表，与点集一一对应
    :param missing_rate: 各要素类型的缺失率
    :return:
    """
    result_dict = {}
    result_label_dict = {}

    for key, points in point_dict.items():
        labels = pattern_labels.get(key, [])
        if len(points) == 0:
            result_dict[key] = points
            result_label_dict[key] = labels
            continue
        if len(points) != len(labels):
            print(f"警告: {key} 的点数({len(points)})和标签数({len(labels)})不匹配")

        rate = missing_rate.get(key, 0)
        n_keep = int(len(points) * (1 - rate))
        if n_keep > 0:
            indices = np.random.choice(len(points), n_keep, replace=False)
            selected_points = np.array(points)[indices]
            labels_array = np.array(labels) if not isinstance(labels, np.ndarray) else labels
            result_dict[key] = list([list(temp) for temp in selected_points])
            result_label_dict[key] = labels_array[indices].tolist()
        else:
            result_dict[key] = np.empty((0, 3))
            result_label_dict[key] = []

    return result_dict, result_label_dict


def add_noise_points(point_dict, pattern_labels, noise_rate):
    """
    # Step 5: 添加噪声
    :param point_dict: 各要素类型的点集字典
    :param pattern_labels:  各要素类型的标签列表，与点集一一对应
    :param noise_rate: 每个要素添加的噪声点率
    :return:
    """
    result_dict = {}
    result_label_dict = {}
    for key, points in point_dict.items():
        current_labels = pattern_labels.get(key, [])
        n_noise = int(len(points) * noise_rate)
        if n_noise > 0:
            noise_points = []
            noise_labels = []
            for _ in range(n_noise):
                x = np.random.uniform(*ST_X)
                y = np.random.uniform(*ST_Y)
                noise_points.append((x, y))
                noise_labels.append('noise')  # 噪声点的标签为'noise'

            # 合并原有点和噪声点
            if len(points) > 0:
                points.extend(noise_points)
                result_dict[key] = points
                result_label_dict[key] = current_labels + noise_labels
            else:
                result_dict[key] = noise_points
                result_label_dict[key] = noise_labels
        else:
            result_label_dict[key] = current_labels
            result_dict[key] = points

    return result_dict, result_label_dict


def generate_independent_element(n_points: int, element_name: str, cluster_params: Optional[Dict] = None):
    """
    Step 6: 生成独立的时空要素（使用不同的簇过程参数）
    :param element_name: 要素名称（如'D', 'E'）
    :param n_points:  生成的点数量
    :param cluster_params:  簇过程参数，如果为None则使用均匀随机分布
    :return: 生成的时空点集
    """
    points = []
    if cluster_params is None:
        # 完全随机分布
        for _ in range(n_points):
            x = np.random.uniform(*ST_X)
            y = np.random.uniform(*ST_Y)
            points.append((x, y))
        labels = ['independent'] * n_points
        return points, labels
    else:
        # 使用簇过程生成（与主过程类似但参数不同）
        n_parent = cluster_params.get('n_parent', 5)
        mean_n_child = cluster_params.get('mean_n_child', 5)
        r_child = cluster_params.get('r_child', 15)
        t_child = cluster_params.get('t_child', 15)

        parent_points = generate_parent_points(n_parent)
        child_points = generate_child_points(parent_points, mean_n_child)

        # 如果点数不足，随机补充
        if len(child_points) < n_points:
            n_extra = n_points - len(child_points)
            print('n_extra', n_points, len(child_points))
            x_extra = np.random.uniform(ST_X[0], ST_X[1], n_extra)
            y_extra = np.random.uniform(ST_Y[0], ST_Y[1], n_extra)
            extra_points = np.column_stack([x_extra, y_extra])

            extra_labels = ['independent_extra'] * n_extra
            labels = ['independent'] * len(child_points) + extra_labels
            child_points = np.vstack([child_points, extra_points])
            # print(len(child_points), len(extra_labels), n_points)
        elif len(child_points) > n_points:
            indices = np.random.choice(len(child_points), n_points, replace=False)
            child_points = child_points[indices]
            labels = ['independent'] * n_points
        else:
            labels = ['independent'] * len(child_points)

        # print('independent', len(child_points), len(labels))
        return child_points, labels


def generate_Spatial_cooccurrence_pattern_with_labels(n_parent: int = 10, mean_n_child: float = 10, r_child: float = 15,
                                                      n_grandchild: int = 3, r_grandchild: float = 5,
                                                      missing_rates: Dict[str, float] = None, noise_rate: float = 0.5):
    """
    生成具有预设时空同现模式的完整数据集
    :param n_parent:  父类点数量
    :param mean_n_child:  每个父类点子类点数量的均值
    :param r_child:  子类点空间半径
    :param n_grandchild: 时空要素类型数量
    :param r_grandchild: 孙子类点空间半径（同现关系半径）
    :param missing_rates: 各要素类型的缺失率
    :param noise_rate: 每个要素添加的噪声点数量
    :return:各时空要素的点集字典
    """
    # 步骤1: 生成父类点
    parent_points = generate_parent_points(n_parent)

    # 步骤2: 生成子类点
    child_points = generate_child_points(parent_points, mean_n_child)

    # 步骤3: 生成孙子类点（不同要素类型）
    point_dict, pattern_labels = generate_grandchild_points_with_labels(child_points, n_grandchild,
                                                                        r_grandchild)
    # 步骤4: 移除缺失点
    if missing_rates is not None:
        point_dict, pattern_labels = remove_missing_points(point_dict, pattern_labels, missing_rates)

    # 步骤5: 添加噪声点

    point_dict, pattern_labels = add_noise_points(point_dict, pattern_labels, noise_rate)
    # 重命名要素类型
    # element_names = {f'type_{i}': name for i, name in enumerate(['A', 'B', 'C'])}
    # point_dict = {element_names.get(k, k): v for k, v in point_dict.items()}

    # 生成独立的时空要素 D 和 E
    print("\n生成独立的时空要素 D 和 E...")
    # 要素 D：使用不同参数的簇过程
    d_params = {
        'n_parent': 8,
        'mean_n_child': 8,
        'r_child': 20,
        't_child': 20
    }
    element_D_points, element_D_labels, = generate_independent_element(n_points=80, element_name='D',
                                                                       cluster_params=None)

    # 要素 E：完全随机分布
    element_E_points, element_E_labels = generate_independent_element(n_points=80, element_name='E',
                                                                      cluster_params=None)
    # 将D和E添加到字典中
    point_dict['D'] = element_D_points
    pattern_labels['D'] = element_D_labels

    point_dict['E'] = element_E_points
    pattern_labels['E'] = element_E_labels

    # 转换为DataFrame
    all_data = []
    type_mapping = {'type_0': 'A', 'type_1': 'B', 'type_2': 'C'}

    for elem_type, points in point_dict.items():
        if len(points) == 0:
            continue
        df = pd.DataFrame(points, columns=['x', 'y'])
        # 映射要素类型名称
        if elem_type in type_mapping:
            df['element_type'] = type_mapping.get(elem_type)
        else:
            df['element_type'] = elem_type

        # df['element_type'] = elem_type
        print(len(df), len(pattern_labels[elem_type]), elem_type)
        df['pattern_id'] = pattern_labels[elem_type]
        all_data.append(df)

    result_df = pd.concat(all_data, ignore_index=True)

    # 将pattern_id映射为具体的模式名称
    # 同一个child_id下的所有类型点构成一个同现模式
    pattern_mapping = {}
    for elem_type, labels in pattern_labels.items():
        for label in labels:
            if label not in pattern_mapping and label not in ['noise', 'independent_extra', 'independent']:
                # 提取child_id

                child_id = label.split('_')[1]
                pattern_mapping[label] = f'pattern_{child_id}'

    result_df['pattern_name'] = result_df['pattern_id'].map(
        lambda x: pattern_mapping.get(x, x)
    )

    return point_dict, result_df


def plot_3d(data):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = {
        'A': 'r',
        'B': 'g',
        'C': 'b',
        'D': 'orange',
        'E': 'purple'
    }

    for k, pts in data.items():
        pts = np.array(pts)
        if k in ['D', 'E']:
            continue
        if len(pts) == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=colors.get(k, 'black'), label=k)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('T')
    ax.legend()
    plt.show()


def simulate_spatial_co_locations():
    print("生成具有预设时空同现模式的数据...")

    cooccurrence_data, cooccurrence_df = generate_Spatial_cooccurrence_pattern_with_labels(n_parent=n_parent,
                                                                                           mean_n_child=mean_n_child,
                                                                                           r_child=r_child,
                                                                                           n_grandchild=n_grandchild,
                                                                                           r_grandchild=r_grandchild,
                                                                                           missing_rates=missing_rates,
                                                                                           noise_rate=noise_rate)

    cooccurrence_df.to_csv('spatial_co_location.csv', index=False)

    return cooccurrence_data


if __name__ == "__main__":
    data = simulate_spatial_co_locations()
    plot_3d(data)

    # 输出统计
    print("===== 数据统计 =====")
    for k, v in data.items():
        print(f"{k}: {len(v)} points")
