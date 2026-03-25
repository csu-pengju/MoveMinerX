#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: data_manager.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 14:22
from moveminerx.data_management.loader.factory import get_loader
from moveminerx.data_management.dataset.dataset import Dataset
from moveminerx.data_management.dataset.registry import load_dataset as load_builtin


def load_data(path=None,
              data_type="trajectory",
              dataset_name=None,
              **kwargs):
    """
    统一数据加载入口（推荐用户使用这个）

    参数：
    - path: 文件路径
    - data_type: point / trajectory / flow
    - dataset_name: benchmark名称
    """

    # 1️⃣ 优先加载内置数据集
    if dataset_name is not None:
        return load_builtin(dataset_name, **kwargs)

    if path is None:
        raise ValueError("Either path or dataset_name must be provided")

    # 2️⃣ 自动选择loader
    loader = get_loader(path)

    # 3️⃣ 根据数据类型加载
    if data_type == "point":
        data = loader.load_points(path, **kwargs)

    elif data_type == "trajectory":
        data = loader.load_trajectories(path, **kwargs)

    elif data_type == "flow":
        data = loader.load_flows(path, **kwargs)

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    return Dataset(data, data_type=data_type, name=path)