#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: shapefile_loader.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 13:56
import geopandas as gpd
from .base_loader import BaseLoader
from moveminerx.data_management.core.point import MovingPoint


class ShapefileLoader(BaseLoader):

    def load_points(self, path, id_col="id", t_col=None):
        gdf = gpd.read_file(path)

        points = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            points.append(
                MovingPoint(
                    row[id_col],
                    geom.x,
                    geom.y,
                    row[t_col] if t_col else None
                )
            )

        return points

    def load_trajectories(self, path, **kwargs):
        raise NotImplementedError("Shapefile trajectory not standard")

    def load_flows(self, path, **kwargs):
        raise NotImplementedError