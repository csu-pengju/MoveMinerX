#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _CPMAlgorithm.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2025/11/19 14:05
import copy
import math
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.neighbors import BallTree

from moveminerx.mining.convergence._basic_class import MovingObject, TrajectoryPoint, SnapshotBuilder, ConvergingMPattern, \
    TrajectoryLoader


class CPMMiner:

    def __init__(self, objects: Dict[str, MovingObject], dc: float = 1000.0, rho_threshold: int = 20,
                 delta_threshold: float = 1500.0, epsilon: float = 800.0,
                 r: float = 5000.0, s_threshold: int = 5, k1: int = 20, k2: int = 5):

        # 算法参数
        self.dc = dc  # 密度计算半径
        self.rho_threshold = rho_threshold  # 密度阈值
        self.delta_threshold = delta_threshold  # 距离阈值
        self.epsilon = epsilon  # 邻域半径
        self.r = r  # 汇聚区域半径  Nr(pa), pa为某汇聚中心点
        self.s_threshold = s_threshold  # 同向点集阈值
        self.k1 = k1  # 时间持续性阈值
        self.k2 = k2  # 最小持续时间阈值

        # 数据存储
        self.objects = objects
        self.snapshots = SnapshotBuilder.generate_snapshots(objects=objects)
        self.timestamps = list(self.snapshots.keys())
        self.timestamps.sort()

    def _calculate_density_peaks(self, points: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """计算密度峰值点 - 算法1: DPQ"""
        n = len(points)
        if n == 0:
            return []

        # 构建空间索引
        coordinates = np.array([[point.x, point.y] for point in points])
        tree = BallTree(coordinates)

        # 计算每个点的密度
        densities = []
        for i, point in enumerate(points):
            indices = tree.query_radius([coordinates[i]], r=self.dc)[0]
            density = len(indices)
            densities.append(density)

        # 计算每个点到更高密度点的最小距离
        deltas = []
        for i in range(n):
            if densities[i] == max(densities):
                # 密度最大的点，取到所有点的最大距离
                max_dist = 0
                for j in range(n):
                    if i != j:
                        dist = points[i].distance_to(points[j])
                        max_dist = max(max_dist, dist)
                deltas.append(max_dist)
            else:
                # 找到密度更高的点中的最小距离
                min_dist = float('inf')
                for j in range(n):
                    if densities[j] > densities[i]:
                        dist = points[i].distance_to(points[j])
                        min_dist = min(min_dist, dist)
                deltas.append(min_dist if min_dist != float('inf') else 0)

        # 筛选密度峰值点
        peak_points = []
        for i, point in enumerate(points):
            if densities[i] > self.rho_threshold and deltas[i] > self.delta_threshold:
                peak_points.append(point)

        return peak_points

    def _get_neighborhood(self, center_point: TrajectoryPoint, timestamp: int, radius: float) -> List[TrajectoryPoint]:
        """获取指定中心点邻域内的点"""
        snapshot = self.snapshots[timestamp]
        neighborhood = []

        for point in snapshot.points.values():
            if center_point.distance_to(point) <= radius:
                neighborhood.append(point)

        return neighborhood

    def _get_direction_regions(self, center_point: TrajectoryPoint, points: List[TrajectoryPoint],
                               n_directions: int = 4) -> Dict[int, List[TrajectoryPoint]]:
        """将点划分到方向区域"""
        regions = {i: [] for i in range(n_directions)}

        for point in points:
            # 计算相对坐标
            dx = point.x - center_point.x
            dy = point.y - center_point.y

            # 计算极角
            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += 2 * math.pi

            # 确定方向区域
            region_idx = int(angle / (2 * math.pi / n_directions))
            regions[region_idx].append(point)

        return regions

    def _get_direction_region(self, p_curr: TrajectoryPoint, p_center: TrajectoryPoint, n_directions: int = 4) -> int:
        """
        计算当前点 p_curr 相对于汇聚中心 p_center 的方向区域 D_i。
        该函数实现公式 (1) 或图2中的扇形区域划分。 [cite: 101, 157]
        返回区域索引 i (例如 1 到 N_DIRECTIONS)。
        """
        dx = p_center.x - p_curr.x
        dy = p_center.y - p_curr.y
        # 计算极坐标角度 θ = arctan(y/x)
        # math.atan2(dy, dx) 返回的角度范围是 (-π, π]
        angle = math.atan2(dy, dx)  # [cite: 124] (这里的 dx, dy 定义可能需要调整，以匹配论文中的极坐标定义)
        # 规范化到 [0, 2π)
        if angle < 0:
            angle += 2 * math.pi

        # 划分区域： (2i-3)π/n <= θ < (2i-1)π/n [cite: 101]
        sector_angle = 2 * math.pi / n_directions
        region_index = int(angle // sector_angle) + 1  # 1-based indexing for region D_i

        return region_index

    def _is_converging(self, obj: MovingObject, center_point: TrajectoryPoint, end_time: int, k1: int) -> bool:
        """
        检查移动对象 o 在时间窗 [t_e - k_1 + 1, t_e] 内是否满足汇聚条件。
        即：对象在过去 k1 个时间戳内都在向 p_center 移动。 [cite: 133, 144]
        """
        # 获取汇聚组 A' 在过去 k1 个时间戳上的空间点 S' [cite: 133]
        start_time = end_time - k1 + 1  # t_{e - k_1 + 1}
        # 历史轨迹点集合
        past_points: List[TrajectoryPoint] = []
        for t in range(start_time, end_time):
            if t in obj.trajectory.points:
                past_points.append(obj.trajectory.get_point(t))
        current_point = obj.trajectory.get_point(end_time)

        if not current_point or not past_points:
            return False  # 数据不足，无法判断

        # 检查方向性：所有历史点都必须位于 p_center 不同的方向区域，且距离递减
        # 由于距离递减/方向区域变化的判断较为复杂，这里只实现一个简化的方向性检查：
        # 假设汇聚要求所有点在过去 k1-1 个时间间隔内都位于朝向 p_center 的方向区域
        # 根据论文公式4，检查 k1 个时间戳内，对象是否在中心区域 N_R(p_a) 外部向其靠近 [cite: 124]
        # 简化实现：检查过去 k1 个时间戳内，点到中心的距离是否总体呈下降趋势
        distances = [p.distance_to(center_point) for p in past_points] + [current_point.distance_to(center_point)]

        if len(distances) < k1:
            return False  # 历史数据点不足 k1 个

        # 检查所有距离是否都大于 r（汇聚区域半径），且距离递减
        for i in range(len(distances)):
            dist = distances[i]

            # 条件 1：点必须在汇聚中心区域 N_R(p_a) 外部 [cite: 124]
            if dist < R:
                return False

                # 条件 2：距离递减 (简化为相邻距离比较)
            if i > 0 and dist >= distances[i - 1]:
                # 允许距离不变，但不能增加（严格汇聚）
                return False

                # 满足所有 k1 个时间戳都在中心区域外，且距离递减
        return True

    def _obtain_converging_group(self, center_point: TrajectoryPoint, t: int) -> List[TrajectoryPoint]:
        # 根据定义5 获取汇聚群体
        outer_points = self._get_neighborhood(center_point, max(1, t - self.k1), self.r)
        inner_points = self._get_neighborhood(center_point, t, self.epsilon)

        current_converging_points = [p for p in inner_points
                                     if self.objects.get(p.oid).trajectory.get_point(t - self.k1) not in outer_points]
        # print('dsdsd')
        # print(outer_points)
        for p in inner_points:
            last_p = self.objects.get(p.oid).trajectory.get_point(t - self.k1)
            # print(last_p in outer_points)
            # print(p, last_p)

        return current_converging_points

    def _is_converging_group(self, center_point: TrajectoryPoint, current_t: int) -> bool:
        """根据定义5、6判断是否构成向心汇聚群体"""
        # 所有历史点都必须位于p_center不同的方向区域
        # 获取r半径邻域内的点

        current_converging_points = self._obtain_converging_group(center_point, current_t)
        if current_t - 1 > 0:
            last_converging_points = self._obtain_converging_group(center_point, current_t - 1)
        else:
            last_converging_points = []
        if len(current_converging_points) == 0:
            return False
        start_time = max(1, current_t - self.k1 + 1)  # t_{e - k_1 + 1}
        # 根据定义6，获取向心汇聚群体
        if len(last_converging_points) == 0:
            diff_converging_points = copy.copy(current_converging_points)
        else:
            diff_converging_points = [p for p in current_converging_points if p not in last_converging_points]
        # print(current_t, 'converging_points', diff_converging_points)

        for t in range(start_time, current_t):
            history_points = [self.objects.get(p.oid).trajectory.get_point(t) for p in diff_converging_points]
            direction_regions = self._get_direction_regions(center_point, history_points)
            # 划分方向区域
            # print('direction_regions', direction_regions)
            # 检查每个方向区域是否满足阈值
            for region_points in direction_regions.values():
                if len(region_points) < self.s_threshold:
                    return False

            return True

    def mine_converging_patterns(self, end_time: int=None) -> List[ConvergingMPattern]:
        """挖掘汇聚模式 - 主算法"""
        # print("开始挖掘汇聚模式...")
        patterns = []
        candidate_patterns = {}  # center -> ConvergingPattern
        # 按时间顺序处理
        if len(self.timestamps) < 1:
            return []
        if not end_time:
            end_time = self.timestamps[-1]

        for timestamp in self.timestamps[:end_time+1]:
            snapshot = self.snapshots[timestamp]
            # print(snapshot.points.values())
            # 1. 定位密度峰值点
            density_peaks = self._calculate_density_peaks(list(snapshot.points.values()))
            # print('density_peaks', density_peaks)
            # 2. 检查每个候选中心点
            current_candidates = {}
            # print('density_peaks: ', len(density_peaks))
            used_peak_point = set()
            for peak_point in density_peaks:
                if peak_point.oid in used_peak_point:
                    continue
                center = (peak_point.x, peak_point.y)
                # 检查是否构成向心汇聚
                is_converging_group = self._is_converging_group(peak_point, timestamp)

                if is_converging_group:
                    # 更新或创建候选模式         # 汇聚中心不是停止了
                    existing_pattern = self._find_existing_pattern_for_center(center, list(candidate_patterns.values()),
                                                                              timestamp)
                    # print('existing_pattern', existing_pattern)
                    if existing_pattern:
                        # print('existing ', existing_pattern)
                        # if center in candidate_patterns:
                        pattern = copy.copy(candidate_patterns[existing_pattern.center])
                        object_ids = {p.oid for p in self._get_neighborhood(peak_point, timestamp, self.r)}
                        for oid in object_ids:
                            used_peak_point.add(oid)
                        # print('objectos', object_ids)
                        pattern.add_group(timestamp, object_ids)
                        pattern.center = center
                        pattern.end_time = timestamp
                        # candidate_patterns[center] = pattern
                        current_candidates[center] = pattern
                        del candidate_patterns[existing_pattern.center]
                        if existing_pattern.center in current_candidates:
                            del current_candidates[existing_pattern.center]

                        # current_candidates[center]=candidate_patterns[center]
                    else:
                        pattern = ConvergingMPattern(center, timestamp, timestamp)
                        object_ids = {p.oid for p in self._get_neighborhood(peak_point, timestamp, self.r)}
                        for oid in object_ids:
                            used_peak_point.add(oid)
                        pattern.add_group(timestamp, object_ids)
                        # candidate_patterns[center] = pattern
                        current_candidates[center] = pattern

                    # current_candidates[center] = candidate_patterns[center]
            # 3. 检查完成的模式
            completed_patterns = []
            # print('candidate_patterns', candidate_patterns)
            for center, pattern in candidate_patterns.items():
                # print(timestamp, 'pattern', pattern)
                if pattern.duration() >= self.k2:
                    patterns.append(pattern)
                    completed_patterns.append(center)
                # if center not in current_candidates:
                #     # 候选模式在当前时间不再活跃
                #     if pattern.duration() >= self.k2:
                #         patterns.append(pattern)
                #         completed_patterns.append(center)

            # 移除完成的模式
            for center in completed_patterns:
                del candidate_patterns[center]
            candidate_patterns = current_candidates
            # print('candidate_patterns', candidate_patterns)

        # 添加仍在活跃的模式（如果满足持续时间要求）
        for pattern in candidate_patterns.values():
            if pattern.duration() >= self.k2:
                patterns.append(pattern)

        # 过滤满足k1持续时间的模式
        final_patterns = [p for p in patterns if p.duration() >= self.k1]

        # print(f"发现 {len(final_patterns)} 个汇聚模式")
        return final_patterns

    def _find_existing_pattern_for_center(self, center: Tuple[float, float], patterns: List[ConvergingMPattern],
                                          timestamp: int, spatial_tolerance: float = 200.0) -> Optional[ConvergingMPattern]:

        """为当前中心点查找现有的汇聚模式"""
        for pattern in patterns:
            # 检查空间邻近性和时间连续性
            center1 = pattern.center
            center2 = center
            # print(math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2))
            if (self._is_same_converging_center(pattern.center, center, spatial_tolerance) and
                    pattern.end_time >= timestamp - 2):  # 允许短暂的时间间隔
                return pattern
        return None

    def _is_same_converging_center(self, center1: Tuple[float, float], center2: Tuple[float, float],
                                   tolerance: float = 200.0) -> bool:
        """判断两个中心点是否属于同一个汇聚模式，考虑空间容差"""
        distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance <= tolerance


# 测试代码
def test_cpm_algorithm():
    """测试CPM算法"""
    # 创建算法实例

    objects = TrajectoryLoader.load_from_shp(rf'..\simulated Ex\data\D4.shp')
    # D2
    # cpm = CPMMiner(objects=objects, dc=50.0, rho_threshold=3, delta_threshold=10.0,
    #                epsilon=10.0, r=100.0, s_threshold=3, k1=2, k2=5)
    cpm = CPMMiner(objects=objects, dc=200.0, rho_threshold=4, delta_threshold=10.0,
                   epsilon=40.0, r=100.0, s_threshold=0, k1=2, k2=5)
    # 挖掘汇聚模式
    start_time = time.time()
    patterns = cpm.mine_converging_patterns()
    end_time = time.time()

    print(f"算法运行时间: {end_time - start_time:.2f} 秒")

    # 输出结果
    # 输出结果
    for i, pattern in enumerate(patterns):
        print(f"模式 {i + 1}: {pattern}")
        print(f"  持续时间: {pattern.duration()} 时间单位")
        print(f"  涉及对象数: {len(pattern.groups)} 个时间戳")

    return patterns


# if __name__ == "__main__":
#     test_cpm_algorithm()
