#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: OD_extraction.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/5/26 22:18
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from tests.util import read_shapefile

geoLife_path = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\datasets\case studies\GeoLife.shp'
data_folder = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\datasets\case studies'

def OD_extraction():
    geolife = read_shapefile(geoLife_path)
    print(geolife.columns)
    od_records = []
    geos = geolife['geometry'].values.tolist()

    for row in geolife.iterrows():
        # print(row)
        start_point = list(row[1].geometry.coords)[0]
        end_point = list(row[1].geometry.coords)[-1]
        # print(end_point, start_point)
        od_record = {
            'OD_ID': row[1].traj_id,  # OD对ID
            'start_x': start_point[0],
            'start_y': start_point[1],
            'end_x': end_point[0],
            'end_y': end_point[1],
            'geometry': LineString([[start_point[0], start_point[1]], [end_point[0], end_point[1]]])  #用起点作为几何，也可以改为终点或连线
        }
        od_records.append(od_record)

    od_gdf = gpd.GeoDataFrame(od_records, crs=geolife.crs)
    output_path = rf"{data_folder}\OD_lines.shp"
    od_gdf.to_file(output_path, driver='ESRI Shapefile')


if __name__ == "__main__":
    OD_extraction()

