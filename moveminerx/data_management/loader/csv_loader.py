#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: csv_loader.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 13:38
import pandas as pd
from shapely import wkt

from moveminerx.data_management.core.point import MovingPoint
from moveminerx.data_management.core.trajectory import Trajectory
from moveminerx.data_management.core.flow import Flow
from .base_loader import BaseLoader


class CSVLoader:

    def load_points(self, path, id_col="id", x_col="x", y_col="y", t_col="t"):
        df = pd.read_csv(path)

        points = [
            MovingPoint(row[id_col], row[x_col], row[y_col], row[t_col])
            for _, row in df.iterrows()
        ]

        return points

    def load_trajectory(self, path,
                        id_col="id",
                        x_col="x",
                        y_col="y",
                        t_col="t"):
        df = pd.read_csv(path)

        trajectories = []

        for obj_id, group in df.groupby(id_col):
            points = [
                MovingPoint(
                    obj_id,
                    row[x_col],
                    row[y_col],
                    row[t_col]
                )
                for _, row in group.iterrows()
            ]
            trajectories.append(Trajectory(obj_id, points))

        return trajectories

    def load_flows(self, path,
                   mode="od",  # "od" or "wkt"
                   origin_x="ox", origin_y="oy",
                   dest_x="dx", dest_y="dy",
                   wkt_col="geometry",
                   volume_col="volume"):
        df = pd.read_csv(path)
        flows = []

        if mode == "od":
            for i, row in df.iterrows():
                coords = [
                    (row[origin_x], row[origin_y]),
                    (row[dest_x], row[dest_y])
                ]
                flows.append(
                    Flow(coords, flow_id=i,
                         volume=row.get(volume_col, 1))
                )

        elif mode == "wkt":
            for i, row in df.iterrows():
                geom = wkt.loads(row[wkt_col])
                flows.append(
                    Flow(list(geom.coords),
                         flow_id=i,
                         volume=row.get(volume_col, 1))
                )

        return flows
