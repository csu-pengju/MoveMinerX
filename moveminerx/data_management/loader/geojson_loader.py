#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: geojson_loader.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 13:56

import geopandas as gpd
from .base_loader import BaseLoader


class GeoJSONLoader(BaseLoader):

    def load_points(self, path, **kwargs):
        gdf = gpd.read_file(path)
        return gdf

    def load_trajectories(self, path, **kwargs):
        gdf = gpd.read_file(path)
        return gdf

    def load_flows(self, path, **kwargs):
        raise NotImplementedError