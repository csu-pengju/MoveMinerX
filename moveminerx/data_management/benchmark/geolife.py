#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: geolife.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 13:53
from moveminerx.data_management.dataset.registry import register_dataset
from moveminerx.data_management.loader.csv_loader import CSVLoader
from moveminerx.data_management.dataset.dataset import Dataset


@register_dataset("geolife")
def load_geolife(path="data/geolife.csv"):
    loader = CSVLoader()
    trajs = loader.load_trajectory(path)

    return Dataset(trajs, data_type="trajectory", name="Geolife")