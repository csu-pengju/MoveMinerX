#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: factory.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 13:56
import os
from .csv_loader import CSVLoader
from .shapefile_loader import ShapefileLoader
from .geojson_loader import GeoJSONLoader


def get_loader(path):
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".csv":
        return CSVLoader()
    elif ext == ".shp":
        return ShapefileLoader()
    elif ext in [".geojson", ".json"]:
        return GeoJSONLoader()
    else:
        raise ValueError(f"Unsupported format: {ext}")