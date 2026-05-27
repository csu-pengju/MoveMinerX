#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: data_preprocessing.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/4/19 18:15
import pandas as pd
from shapely.geometry import LineString, Point
import geopandas as gpd
import os
import arff


def od_to_shp():
    folder = r'D:\SDBC-master\SDBC-master\SDBC\Data'
    datasets = ['SD1', 'SD2', 'SD3', 'SD4', 'SD5', 'SD6']

    for i in range(1, 8):
        dataset = f'SD{i}'
        df = pd.read_csv(rf'{folder}\{dataset}.csv')
        print(df.columns)
        # 存储线段几何和属性
        lines = []
        attributes = []
        for idx, row in df.iterrows():
            oid = row['id']
            ox = row['ox']
            oy = row['oy']
            dx = row['dx']
            dy = row['dy']
            IDX = int(row['ori_cluster']) - 1
            line = LineString([(ox, oy), (dx, dy)])
            lines.append(line)
            attr = {
                'oid': oid,  # 原始行号
                'ori_cluster': row.get('ori_cluster', -1),  # 原始聚类标签
                'ox': ox,
                'oy': oy,
                'dx': dx,
                'dy': dy,
                'IDX': IDX
            }
            attributes.append(attr)
        gdf = gpd.GeoDataFrame(attributes, geometry=lines, crs='EPSG:4326')  # 假设是WGS84坐标系
        saved_folder = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\datasets\aggreation patterns'
        output_path = os.path.join(saved_folder, f'FD{i}.shp')
        gdf.to_file(output_path, driver='ESRI Shapefile', crs='EPSG:3857')
        print(f"Saved {len(lines)} lines to {output_path}")


def arff_to_shp():
    folder = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\datasets\used'
    files = os.listdir(folder)
    for file in files:
        if file.endswith('.arff'):
            pts = []
            attributes = []
            with open(rf'{folder}\{file}', 'r') as f:
                dataset = arff.load(f)

                data = dataset['data']
                for row in data:
                    x = row[0]
                    y = row[1]
                    if row[2] == 'noise':
                        cluster_class = -1
                    else:
                        cluster_class = int(row[2])
                    pt = Point((x, y))
                    pts.append(pt)
                    attributes.append({'x': x, 'y': y, 'IDX': cluster_class})
            gdf = gpd.GeoDataFrame(attributes, geometry=pts, crs='EPSG:4326')  # 假设是WGS84坐标系
            saved_folder = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\datasets\aggreation patterns'
            filename = file.replace('arff', 'shp')
            output_path = os.path.join(saved_folder, f'PD_{filename}')
            gdf.to_file(output_path, driver='ESRI Shapefile', crs='EPSG:3857')
            print(f"Saved {len(pts)} pts to {output_path}")


def mat_to_shp():
    import scipy.io as sio
    folder = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\datasets\used'
    files = os.listdir(folder)
    for file in files:
        pts = []
        attributes = []
        if file.endswith('.mat'):
            mat_file_path = rf'{folder}\{file}'
            mat_data = sio.loadmat(mat_file_path)
            data = mat_data['data']
            labels = mat_data['label']

            for row, label in zip(data, labels):
                if label[0] == 'noise':
                    cluster_class = -1
                else:
                    cluster_class = int(label[0])
                pt = Point((row[0], row[1]))
                pts.append(pt)
                attributes.append({'x': pt.x, 'y': pt.y, 'IDX': cluster_class})
            gdf = gpd.GeoDataFrame(attributes, geometry=pts, crs='EPSG:4326')  # 假设是WGS84坐标系
            saved_folder = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\datasets\aggreation patterns'
            filename = file.replace('mat', 'shp')
            output_path = os.path.join(saved_folder, f'PD_{filename}')
            gdf.to_file(output_path, driver='ESRI Shapefile', crs='EPSG:3857')
            print(f"Saved {len(pts)} pts to {output_path}")


if __name__ == "__main__":
    mat_to_shp()
