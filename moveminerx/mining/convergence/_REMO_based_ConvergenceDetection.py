#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _REMO_based_ConvergenceDetection.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/10/21 16:17
import itertools
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

# from baselines.basic_class import MovingObject, Trajectory, TrajectoryPoint, REMOConvergence, SnapshotBuilder
from moveminerx.mining.convergence._basic_class import MovingObject,  SnapshotBuilder, TrajectoryPoint, REMOConvergence, REMOConvergence
# --- Snapshot / 半直线集合 ------------------------------------------------
from moveminerx.utils.geometry_utils import intersect_lines, point_is_in_front_of_ray, dist_point_to_infinite_line
from moveminerx.utils.util import calculate_distance, save_shapefile


@dataclass
class Ray:
    obj_id: str
    x0: float
    y0: float
    dx: float
    dy: float


@dataclass
class RayBuffer1:
    """射线缓冲区"""
    obj_id: str
    x0: float
    y0: float
    dx: float
    dy: float
    buffer_radius: float
    buffer_polygon: Optional[Polygon] = None

    def __post_init__(self):
        if self.buffer_polygon is None:
            self.buffer_polygon = self._create_buffer_polygon()

    def _create_buffer_polygon(self) -> Polygon:
        """创建射线缓冲区多边形"""

        # 归一化方向向量
        length = math.sqrt(self.dx * self.dx + self.dy * self.dy)
        if length == 0:
            return Point(self.x0, self.y0).buffer(self.buffer_radius)

        # dx, dy = self.dx / length, self.dy / length

        # 创建射线线段（延长到足够远）
        # end_x = self.x0 + dx * 10000  # 足够长的射线
        # end_y = self.y0 + dy * 10000
        end_x = self.x0 + self.dx * 100  # 足够长的射线
        end_y = self.y0 + self.dy * 100

        # 创建线段并添加缓冲区
        ray_line = LineString([(self.x0, self.y0), (end_x, end_y)])
        buffer_polygon = ray_line.buffer(self.buffer_radius)

        return buffer_polygon

    def contains_point(self, px: float, py: float) -> bool:
        """判断点是否在缓冲区内部"""
        return self.buffer_polygon.contains(Polygon(px, py))

    def intersects_with(self, other: 'RayBuffer1') -> bool:
        """判断两个射线缓冲区是否相交"""
        return self.buffer_polygon.intersects(other.buffer_polygon)

    def intersection_area(self, other: 'RayBuffer1') -> Polygon:
        """计算两个缓冲区的相交区域"""
        return self.buffer_polygon.intersection(other.buffer_polygon)


@dataclass
class RayBuffer:
    """射线缓冲区"""
    obj_id: str
    x0: float
    y0: float
    x1: float
    y1: float
    buffer_radius: float
    buffer_polygon: Optional[Polygon] = None

    def __post_init__(self):
        if self.buffer_polygon is None:
            self.buffer_polygon = self._create_buffer_polygon()

    def _create_buffer_polygon(self) -> Polygon:
        """创建射线缓冲区多边形"""
        # 创建线段并添加缓冲区
        ray_line = LineString([(self.x0, self.y0), (self.x1, self.y1)])
        buffer_polygon = ray_line.buffer(self.buffer_radius)
        return buffer_polygon

    def contains_point(self, px: float, py: float) -> bool:
        """判断点是否在缓冲区内部"""
        return self.buffer_polygon.contains(Polygon(px, py))

    def intersects_with(self, other: 'RayBuffer') -> bool:
        """判断两个射线缓冲区是否相交"""
        return self.buffer_polygon.intersects(other.buffer_polygon)

    def intersection_area(self, other: 'RayBuffer') -> Polygon:
        """计算两个缓冲区的相交区域"""
        return self.buffer_polygon.intersection(other.buffer_polygon)

@dataclass
class Snapshot_:
    t0: float
    t1: float
    rays: List[Ray] = field(default_factory=list)

    def add_ray(self, ray: Ray):
        self.rays.append(ray)


class RayBufferManager:
    """管理多条射线缓冲区"""

    def __init__(self, buffer_radius: float = 50.0):
        self.buffer_radius = buffer_radius
        self.ray_buffers: List[RayBuffer] = []

    def add_ray2(self, obj_id: str, x0: float, y0: float, dx: float, dy: float):
        """添加射线缓冲区"""
        ray_buffer = RayBuffer(obj_id=obj_id, x0=x0, y0=y0, dx=dx, dy=dy, buffer_radius=self.buffer_radius)
        self.ray_buffers.append(ray_buffer)

    def add_ray(self, obj_id: str, x0: float, y0: float, x1: float, y1: float):
        """添加射线缓冲区"""
        ray_buffer = RayBuffer(obj_id=obj_id, x0=x0, y0=y0, x1=x1, y1=y1, buffer_radius=self.buffer_radius)
        self.ray_buffers.append(ray_buffer)

    def add_rays(self, rays: List[Tuple[str, float, float, float, float]]):
        """批量添加射线"""
        for obj_id, x0, y0, dx, dy in rays:
            self.add_ray2(obj_id=obj_id, x0=x0, y0=y0, dx=dx, dy=dy)

    def find_intersection_clusters(self, min_intersection_area: float = 10.0,
                                   min_rays: int = 3) -> List[dict]:
        """
        查找射线缓冲区相交的聚类区域（要求所有射线互相相交）

        Args:
            min_intersection_area: 最小相交面积阈值
            min_rays: 最小射线数量阈值

        Returns:
            相交区域信息列表
        """
        n = len(self.ray_buffers)

        # 步骤1: 构建相交图
        intersection_graph = {i: set() for i in range(n)}
        idx_obj_id_mapping = {i: self.ray_buffers[i].obj_id for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                buffer_i = self.ray_buffers[i]
                buffer_j = self.ray_buffers[j]

                if buffer_i.intersects_with(buffer_j):
                    intersection = buffer_i.intersection_area(buffer_j)
                    # print(i, j, intersection.area)
                    if intersection.area >= min_intersection_area:
                        intersection_graph[i].add(j)
                        intersection_graph[j].add(i)
        # 步骤2: 查找完全子图（clique）
        # print(intersection_graph)
        cliques = self._find_maximal_cliques(intersection_graph, min_rays)
        # print('cliques: ', cliques)
        # 步骤3: 为每个完全子图计算共同相交区域
        clusters = []
        for clique in cliques:
            if len(clique) >= min_rays:
                common_intersection = self._get_common_intersection(clique)
                if common_intersection and common_intersection.area >= min_intersection_area:
                    clusters.append({
                        'ray_indices': [idx_obj_id_mapping[i] for i in clique],
                        'intersection_polygon': common_intersection,
                        'area': common_intersection.area,
                        'centroid': (common_intersection.centroid.x, common_intersection.centroid.y),
                        'num_rays': len(clique)
                    })
        # print('clusters', len(clusters), clusters)
        return clusters

    def _find_maximal_cliques(self, graph: dict, min_size: int) -> List[set]:
        """
        使用Bron-Kerbosch算法查找最大完全子图
        """

        def bron_kerbosch(r, p, x):
            if not p and not x:
                if len(r) >= min_size:
                    cliques.append(r.copy())
                return

            for v in list(p):
                bron_kerbosch(r | {v}, p & graph[v], x & graph[v])
                p.remove(v)
                x.add(v)

        cliques = []
        bron_kerbosch(set(), set(graph.keys()), set())
        return cliques

    def _get_common_intersection(self, ray_indices: set) -> Optional[Polygon]:
        """
        计算多个射线缓冲区的共同相交区域
        """
        if not ray_indices:
            return None

        indices = list(ray_indices)

        # 从第一个缓冲区开始
        common_intersection = self.ray_buffers[indices[0]].buffer_polygon

        # 逐个求交集
        for i in range(1, len(indices)):
            current_buffer = self.ray_buffers[indices[i]].buffer_polygon
            common_intersection = common_intersection.intersection(current_buffer)

            # 如果没有相交区域，提前返回
            if common_intersection.is_empty:
                return None

        return common_intersection

    def find_intersection_clusters2(self, min_intersection_area: float = 50.0,
                                   min_rays: int = 2) -> List[dict]:
        """
        查找射线缓冲区相交的聚类区域
        Args:
            min_intersection_area: 最小相交面积阈值
            min_rays: 最小射线数量阈值

        Returns:
            相交区域信息列表
        """
        n = len(self.ray_buffers)
        intersection_regions = []

        # 检查所有射线对之间的相交
        for i in range(n):
            for j in range(i + 1, n):
                buffer_i = self.ray_buffers[i]
                buffer_j = self.ray_buffers[j]

                if buffer_i.intersects_with(buffer_j):
                    intersection = buffer_i.intersection_area(buffer_j)

                    if intersection.area >= min_intersection_area:
                        intersection_regions.append({
                            'ray_indices': [i, j],
                            'intersection_polygon': intersection,
                            'area': intersection.area,
                            'centroid': (intersection.centroid.x, intersection.centroid.y)
                        })

        # 合并重叠的相交区域
        merged_clusters = self._merge_intersection_clusters(intersection_regions, min_rays)
        return merged_clusters

    def _merge_intersection_clusters(self, intersections: List[dict], min_rays: int) -> List[dict]:
        """合并重叠的相交区域"""
        if not intersections:
            return []

        # 构建相交图
        intersection_graph = {}
        used = set()
        for i, inter1 in enumerate(intersections):
            connected = set(inter1['ray_indices'])
            intersection_graph[i] = connected

            for j, inter2 in enumerate(intersections):
                if i != j:
                    # 如果两个相交区域共享射线或有空间重叠
                    shared_rays = set(inter1['ray_indices']) & set(inter2['ray_indices'])
                    spatial_overlap = inter1['intersection_polygon'].intersects(
                        inter2['intersection_polygon'])

                    if shared_rays or spatial_overlap:
                        connected.update(inter2['ray_indices'])

        # 查找连通分量
        visited = set()
        clusters = []

        for i in range(len(intersections)):
            if i not in visited:
                # 广度优先搜索找连通分量
                cluster_rays = set()
                stack = [i]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster_rays.update(intersection_graph[current])
                        stack.extend([neighbor for neighbor in intersection_graph
                                      if neighbor not in visited and
                                      intersection_graph[current] & intersection_graph[neighbor]])

                if len(cluster_rays) >= min_rays:
                    # 找到该聚类中的所有相交区域
                    cluster_intersections = [intersections[idx] for idx in visited
                                             if idx in range(len(intersections))]

                    # 合并多边形
                    polygons = [inter['intersection_polygon'] for inter in cluster_intersections]
                    merged_polygon = unary_union(polygons)

                    clusters.append({
                        'ray_indices': list(cluster_rays),
                        'intersection_polygon': merged_polygon,
                        'area': merged_polygon.area,
                        'centroid': (merged_polygon.centroid.x, merged_polygon.centroid.y),
                        'num_rays': len(cluster_rays),
                        'num_intersections': len(cluster_intersections) })


        return clusters

    def find_small_intersection_regions(self, max_area: float = 50.0,
                                        min_rays: int = 3) -> List[dict]:
        """
        查找小区域内的多射线相交
        Args:
            max_area: 最大相交区域面积
            min_rays: 最小射线数量

        Returns:
            小相交区域列表
        """
        clusters = self.find_intersection_clusters(min_intersection_area=0.1,
                                                   min_rays=min_rays)

        small_regions = []
        for cluster in clusters:
            if cluster['area'] <= max_area and cluster['num_rays'] >= min_rays:
                small_regions.append(cluster)

        # 按面积排序
        small_regions.sort(key=lambda x: x['area'])

        return small_regions

    def get_rays_in_region(self, region_polygon: Polygon) -> List[int]:
        """获取在指定区域内的射线索引"""
        rays_in_region = []

        for i, ray_buffer in enumerate(self.ray_buffers):
            if region_polygon.intersects(ray_buffer.buffer_polygon):
                rays_in_region.append(i)

        return rays_in_region


# --- Convergence Detector -----------------------------------------------
class ConvergenceDetector:
    """
    Convergence 检测器。对每个时间段 [t, t+interval_length]：
        - 为每个对象拟合方向半直线（ray）
        - 生成候选圆心（采用两两直线交点）
        - 对每个候选圆心统计满足：点位于该对象半直线"前方" 且该点到直线距离 <= r
        - 如果计数 >= m 则标记为 convergence（记录相关信息）

    方法：
        - detect_convergence_serial(moving_objects, t_start, t_end, interval_length, r, m_threshold)
        - detect_convergence_parallel(...)  使用 multiprocessing 并行化时间段处理
    """

    def __init__(self, objects: Dict[str, MovingObject]):
        self.objects = objects
        self.convergences: List[REMOConvergence] = []
        self.min_support: int = 3
        self.radius: float = 200
        self.pattern_counter = 0
        self.snapshots = SnapshotBuilder.generate_snapshots(objects)

    def _build_snapshot(self, t0: int, t1: int) -> Snapshot_:
        s = Snapshot_(t0, t1)
        for mo in self.objects.values():
            seg = mo.get_interval_segment(t0, t1)
            if seg is None:
                continue
            x0, y0, dx, dy = seg
            s.add_ray(Ray(mo.oid, x0, y0, dx, dy))
            # p0 = mo.trajectory.get_point(t0)
            # p1 = mo.trajectory.get_point(t0)
            # s.add_ray(Ray(mo.oid, p0.x, p0.y, p1.x, p1.y))
        return s

    def _snapshot_convergence(self, snapshot: Snapshot_, r: float, m_threshold: int) -> List[Dict]:
        """
        在单个 snapshot 上检测 convergence
        返回匹配到的 pattern 列表，每个 pattern 为 dict：
          { 'center': (x,y), 't0':..., 't1':..., 'members': [obj_ids], 'count': k }
        """
        rays = snapshot.rays
        n = len(rays)
        patterns = []
        if n < m_threshold:
            return patterns

        # 候选圆心：两两直线（无限延拓）交点
        candidate_centers = set()
        count = 0
        for r1, r2 in itertools.combinations(rays, 2):
            ip = intersect_lines(r1.x0, r1.y0, r1.dx, r1.dy, r2.x0, r2.y0, r2.dx, r2.dy)
            # print(ip)
            if ip:
                candidate_centers.add((round(ip[0], 10), round(ip[1], 10)))  # 用 round 降低重复候选

        # 另外：也可以考虑每条 ray 前方某些投影点或 ray 与 ray 形成的小三角形内点作为候选
        # （此处仅使用交点以保持实现清晰；可在未来扩充）
        # print('candidate centers', len(candidate_centers))
        # print(candidate_centers)
        used_rays = set()
        # 对每个候选圆心计数
        for cx, cy in candidate_centers:
            members = []
            for ray in rays:
                # 首先检查圆心是否在 ray 的“前方”
                if ray.obj_id in used_rays:
                    continue
                if not point_is_in_front_of_ray(cx, cy, ray.x0, ray.y0, ray.dx, ray.dy):
                    continue
                    # 其次检查圆心到直线的最短距离是否 <= r
                d = dist_point_to_infinite_line(cx, cy, ray.x0, ray.y0, ray.dx, ray.dy)

                if d <= r + 1e-9:
                    members.append(ray.obj_id)
                    used_rays.add(ray.obj_id)

            if len(members) >= m_threshold:
                patterns.append({
                    'center': (cx, cy),
                    't0': snapshot.t0,
                    't1': snapshot.t1,
                    'members': members,
                    'count': len(members)
                })

        # 去重（简单）：按 center 与成员集合去重
        # 去重：按照成员集合与时间区间，因为可能多个目标相交的区域可以通过多个center+r覆盖
        unique = {}
        for p in patterns:
            # key = (p['center'], tuple(sorted(p['members'])))
            key = (p['t0'], p['t1'], tuple(sorted(p['members'])))
            if key not in unique:
                unique[key] = p
            else:
                # 保留 count 更大的（应该一样）
                if p['count'] > unique[key]['count']:
                    unique[key] = p
        return list(unique.values())

    def _ray_buffer_convergence(self, manager: RayBufferManager, r: float, m_threshold: int) -> List[Dict]:
        """
        在单个 snapshot 上检测 convergence
        返回匹配到的 pattern 列表，每个 pattern 为 dict：
          { 'center': (x,y), 't0':..., 't1':..., 'members': [obj_ids], 'count': k }
        """
        rays = manager.ray_buffers
        n = len(rays)
        patterns = []
        if n < m_threshold:
            return patterns
        intersected_rays = manager.find_intersection_clusters(min_rays=m_threshold,)
        # print('areas', len(intersected_rays))
        return intersected_rays

        # print(areas)

        # # 候选圆心：两两直线（无限延拓）交点
        # candidate_centers = set()
        # count = 0
        # for r1, r2 in itertools.combinations(rays, 2):
        #     ip = intersect_lines(r1.x0, r1.y0, r1.dx, r1.dy, r2.x0, r2.y0, r2.dx, r2.dy)
        #     print(ip)
        #     if ip:
        #         candidate_centers.add((round(ip[0], 10), round(ip[1], 10)))  # 用 round 降低重复候选
        #
        # # 另外：也可以考虑每条 ray 前方某些投影点或 ray 与 ray 形成的小三角形内点作为候选
        # # （此处仅使用交点以保持实现清晰；可在未来扩充）
        # print('candidate centers', len(candidate_centers))
        # print(candidate_centers)
        # used_rays = set()
        # # 对每个候选圆心计数
        # for cx, cy in candidate_centers:
        #     members = []
        #     for ray in rays:
        #         # 首先检查圆心是否在 ray 的“前方”
        #         if ray.obj_id in used_rays:
        #             continue
        #         if not point_is_in_front_of_ray(cx, cy, ray.x0, ray.y0, ray.dx, ray.dy):
        #             continue
        #             # 其次检查圆心到直线的最短距离是否 <= r
        #         d = dist_point_to_infinite_line(cx, cy, ray.x0, ray.y0, ray.dx, ray.dy)
        #
        #         if d <= r + 1e-9:
        #             members.append(ray.obj_id)
        #             used_rays.add(ray.obj_id)
        #
        #     if len(members) >= m_threshold:
        #         patterns.append({
        #             'center': (cx, cy),
        #             't0': snapshot.t0,
        #             't1': snapshot.t1,
        #             'members': members,
        #             'count': len(members)
        #         })
        #
        # # 去重（简单）：按 center 与成员集合去重
        # # 去重：按照成员集合与时间区间，因为可能多个目标相交的区域可以通过多个center+r覆盖
        # unique = {}
        # for p in patterns:
        #     # key = (p['center'], tuple(sorted(p['members'])))
        #     key = (p['t0'], p['t1'], tuple(sorted(p['members'])))
        #     if key not in unique:
        #         unique[key] = p
        #     else:
        #         # 保留 count 更大的（应该一样）
        #         if p['count'] > unique[key]['count']:
        #             unique[key] = p
        # return list(unique.values())

    def detect_convergence_serial2(self, t_start: int, t_end: int, interval_length: int, r: float, m_threshold: int):
        """
        串行检查所有时间区间
        返回 pattern 列表（每个包含 center, t0, t1, members, count）。
        时间段按 [t, t+interval_length] 以步长 interval_length 滑动（不重叠）。
        """
        self.min_support = m_threshold
        self.radius = r
        patterns_all = []
        t = t_start
        while t + interval_length <= t_end + 1e-9:
            snapshot = self._build_snapshot(t, t + interval_length)

            p = self._snapshot_convergence(snapshot, r, m_threshold)
            for temp in p:
                remoConvergence = REMOConvergence(pattern_id=self._pattern_counter(), center=temp['center'],
                                                  members=set(temp['members']), t_start=temp['t0'], t_end=temp['t1'])
                remoConvergence.obj_center = self.calculate_obj_center(temp['members'], time=t + interval_length)
                self.convergences.append(remoConvergence)
            patterns_all.extend(p)
            t += interval_length
        return self.convergences

    def _build_ray_manager(self, t0: int, t1: int) -> RayBufferManager:
        manager = RayBufferManager(self.radius)
        for mo in self.objects.values():
            # print(dir(mo))
            # print('moving objects', mo.get_interval_segment(t0, t1))

            seg = mo.get_interval_segment(t0, t1)
            if seg is None:
                continue
            x0, y0, dx, dy = seg
            # manager.add_ray(mo.oid, x0, y0, dx, dy)
            p0 = mo.trajectory.get_point(t0)
            p1 = mo.trajectory.get_point(t1)
            manager.add_ray(mo.oid, p0.x, p0.y, p1.x, p1.y)

        return manager

    def save_buffers(self, manager: RayBufferManager, dataset):
        saved_folder = rf'D:\Projects\ConvergenceDivergencePatternMining\simulated Ex\results\baseline_shps'
        ray_buffers = manager.ray_buffers
        geometries = [ray_buffer.buffer_polygon for ray_buffer in ray_buffers]
        import geopandas as gpd
        gdf = gpd.GeoDataFrame({'pid': [i+1 for i in range(len(ray_buffers))], 'geometry': geometries},
                               geometry='geometry', crs='EPSG:3857')
        save_shapefile(rf'{saved_folder}\{dataset}_buffer.shp', gdf)

    def detect_convergence_serial(self, t_start: int, t_end: int, interval_length: int, r: float, m_threshold: int,
                                  dataset: str='D3'):
        """
        串行检查所有时间区间
        返回 pattern 列表（每个包含 center, t0, t1, members, count）。
        时间段按 [t, t+interval_length] 以步长 interval_length 滑动（不重叠）。
        """
        self.min_support = m_threshold
        self.radius = r

        t = t_start
        while t + interval_length <= t_end + 1e-9:
            manager = self._build_ray_manager(t, t + interval_length)

            intersected_rays = self._ray_buffer_convergence(manager, r, m_threshold)

            # print('p', intersected_rays)
            for temp in intersected_rays:
                remoConvergence = REMOConvergence(pattern_id=self._pattern_counter(), center=temp['centroid'],
                                                  members=set(temp['ray_indices']), t_start=t, t_end=t+interval_length)

                remoConvergence.obj_center = self.calculate_obj_center(temp['ray_indices'], time=t + interval_length)
                self.convergences.append(remoConvergence)
            # patterns_all.extend(p)
            t += interval_length
            self.save_buffers(manager, dataset=dataset)
        return self.convergences

    def _pattern_counter(self):
        self.pattern_counter += 1
        return self.pattern_counter

    def calculate_obj_center(self, objs, time=1):
        moving_points = self.snapshots[time].points
        points = [[moving_points[obj].x, moving_points[obj].y] for obj in objs]
        x_coords, y_coords = zip(*points)
        centroid_x = sum(x_coords) / len(points)
        centroid_y = sum(y_coords) / len(points)
        return (centroid_x, centroid_y)

    def handled_discovered_convergences2(self):
        # 合并在当前时刻挖掘到的convergences
        processed_convergences = []
        used_pattern_ids = set()
        keys = list(self.convergences.keys())
        for idx1 in range(len(keys)):
            convergence1 = self.convergences[keys[idx1]]

            if idx1 in used_pattern_ids:
                continue
            # print(idx1, 'starting----')
            new_convergence = convergence1
            new_convergence.pattern_id = self._pattern_counter()
            # print('----------', idx1, len(used_pattern_ids))
            for _ in range(0, 3):
                for idx2 in range(idx1 + 1, len(keys)):
                    convergence2 = self.convergences[keys[idx2]]
                    if idx2 in used_pattern_ids:
                        continue
                    obj_center1 = new_convergence.obj_center
                    members1 = new_convergence.members
                    obj_center2 = convergence2.obj_center
                    members2 = set(convergence2.members)
                    center_dist = calculate_distance(obj_center1[0], obj_center1[1], obj_center2[0], obj_center2[1])
                    # print(idx1, idx2, members1, members2, center_dist)
                    # print('members1', members1, 'members2', members2)
                    if len(members2.intersection(members1)) >= self.min_support:
                        # if len(members2.intersection(members1)) > 0 and center_dist < self.radius * 5:
                        # if center_dist < self.radius*2:
                        if True:
                            # print(idx1, idx2, 'has merged ?')
                            new_convergence.update_by_another(convergence2)
                            # del convergence1
                            # convergence1 = new_convergence
                            used_pattern_ids.add(idx2)
                            # print(new_convergence)
            # print(idx1, new_convergence)
            processed_convergences.append(new_convergence)
            used_pattern_ids.add(idx1)
            del new_convergence

        # 按照成员数量排序
        # sorted_convergences = processed_convergences
        sorted_convergences = sorted(processed_convergences,
                                     key=lambda x: len(x.members), reverse=True)
        for idx1 in range(len(sorted_convergences)):
            for idx2 in range(idx1 + 1, len(sorted_convergences)):
                sorted_convergences[idx2].members = sorted_convergences[idx2].members - sorted_convergences[
                    idx1].members

        for idx in range(len(sorted_convergences) - 1, -1, -1):
            if len(sorted_convergences[idx].members) < self.min_support:
                del sorted_convergences[idx]

        return sorted_convergences

    def handled_discovered_convergences(self):
        # 合并在当前时刻挖掘到的convergences
        sorted_convergences = sorted(list(self.convergences.values()),
                                     key=lambda x: len(x.members), reverse=True)
        for idx1 in range(len(sorted_convergences)):
            for idx2 in range(idx1 + 1, len(sorted_convergences)):
                # sorted_convergences[idx2].members = sorted_convergences[idx2].members
                sorted_convergences[idx2].members = sorted_convergences[idx2].members - sorted_convergences[
                    idx1].members

        for idx in range(len(sorted_convergences) - 1, -1, -1):
            if len(sorted_convergences[idx].members) < self.min_support:
                del sorted_convergences[idx]

        return sorted_convergences
