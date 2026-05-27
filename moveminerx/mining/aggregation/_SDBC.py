#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _SDBC.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/5/26 21:46

"""
SDBC (Statistical and Density-Based Clustering) for Geographical Flows
Implementation based on the paper:
"Statistical and density-based clustering of geographical flows for crowd movement patterns recognition"

Author: Implementation based on Tang et al. (2024)
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree, ConvexHull
from typing import List, Tuple, Dict, Set, Optional, Union, Any
from dataclasses import dataclass, field
import geopandas as gpd
from collections import defaultdict
import warnings

from moveminerx.utils.util import read_shapefile


@dataclass
class Flow:
    """Represents an OD (Origin-Destination) flow"""
    flow_id: int
    origin: Tuple[float, float]  # (x, y) or (lon, lat)
    destination: Tuple[float, float]
    timestamp_origin: Optional[float] = None
    timestamp_dest: Optional[float] = None
    attributes: Dict[str, Any] = None

    @property
    def vector(self) -> np.ndarray:
        """Return flow as vector from origin to destination"""
        return np.array([self.destination[0] - self.origin[0],
                         self.destination[1] - self.origin[1]])

    @property
    def length(self) -> float:
        """Calculate flow length"""
        return np.linalg.norm(self.vector)

    def angle_to(self, other: 'Flow') -> float:
        """Calculate angle between this flow and another flow (in degrees)"""
        v1 = self.vector
        v2 = other.vector

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))


class FlowReader:
    """
        Reader for OD flow data from various file formats

        Supported formats:
        - Shapefile (.shp) with line geometries representing flows
        - CSV/Excel files with origin and destination coordinates
        - GeoJSON files
        - DataFrame with OD columns
        """

    def __init__(self):
        self.flows: List[Flow] = []

    def shp_to_flows(self, flow_df: Union[str, pd.DataFrame, 'gpd.GeoDataFrame'],
                     origin_cols: Tuple[str, str] = ('origin_x', 'origin_y'),
                     dest_cols: Tuple[str, str] = ('dest_x', 'dest_y'),
                     id_col: Optional[str] = None,
                     time_origin_col: Optional[str] = None,
                     time_dest_col: Optional[str] = None) -> List[Flow]:

        """
        Convert Shapefile or DataFrame to Flow objects
        Parameters:
        -----------
        flow_df : str, pd.DataFrame, or gpd.GeoDataFrame
        Path to shapefile, DataFrame, or GeoDataFrame containing flow data
        origin_cols : Tuple[str, str]
        column names for origin x and y coordinates
        dest_cols : Tuple[str, str]
        Column names for destination x and y coordinates
        id_col : str, optional
        Column name for flow ID
        time_origin_col : str, optional
        Column name for origin timestamp
        time_dest_col : str, optional
        Column name for destination timestamp

        Returns:
        --------
        List[Flow] : List of Flow objects
        """

        if isinstance(flow_df, gpd.GeoDataFrame):
            return self._geodataframe_to_flows(flow_df, id_col)


        return []

    def _geodataframe_to_flows(self, gdf: 'gpd.GeoDataFrame',
                               id_col: Optional[str] = None,
                               time_origin_col: Optional[str] = None,
                               time_dest_col: Optional[str] = None) -> List[Flow]:

        """Convert GeoDataFrame with line geometries to flows"""
        flows = []
        true_clusters = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            # Extract origin and destination from line geometry
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                if len(coords) >= 2:
                    origin = (coords[0][0], coords[0][1])
                    destination = (coords[-1][0], coords[-1][1])
                else:
                    warnings.warn(f"Line at index {idx} has insufficient coordinates, skipping")
                    continue
            elif geom.geom_type == 'Point':
                # If geometry is point, need separate origin/dest columns
                warnings.warn(
                    "Point geometry detected. Please use DataFrame method with separate origin/dest columns")
                continue
            else:
                warnings.warn(f"Unsupported geometry type: {geom.geom_type}, skipping")
                continue

            # Get flow ID
            if id_col and id_col in row:
                flow_id = row[id_col]
            else:
                flow_id = idx

            # Get timestamps
            time_origin = None
            time_dest = None
            if time_origin_col and time_origin_col in row:
                time_origin = self._parse_timestamp(row[time_origin_col])
            if time_dest_col and time_dest_col in row:
                time_dest = self._parse_timestamp(row[time_dest_col])

            # Collect other attributes
            attributes = {}
            for col in row.index:
                if col not in [id_col, time_origin_col, time_dest_col, 'geometry']:
                    attributes[col] = row[col]

            flow = Flow(
                flow_id=flow_id,
                origin=origin,
                destination=destination,
                timestamp_origin=time_origin,
                timestamp_dest=time_dest,
                attributes=attributes
            )
            flows.append(flow)
            # true_clusters.append(fow)

        self.flows = flows
        return flows

    def _parse_timestamp(self, value) -> Optional[float]:
        """Parse timestamp to numeric value"""
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                # Try to parse as datetime
                from datetime import datetime
                dt = pd.to_datetime(value)
                return dt.timestamp()
            except:
                try:
                    return float(value)
                except:
                    return None

        return None



class SDBC:
    """
    Statistical and Density-Based Clustering for Geographical Flows

    Parameters:
    -----------
    R : float, default=300
        Buffer size for spatial neighborhood (in meters)
    theta : float, default=30
        Direction similarity angle threshold (in degrees)
    lambda_threshold : float, default=0
        G* statistic threshold for high-density flow identification
    alpha : float, default=0.05
        Significance level for Monte Carlo permutation test
    n_permutations : int, default=999
        Number of permutations for significance testing
    """

    def __init__(self, R: float = 300.0, theta: float = 30.0,
                 lambda_threshold: float = 0.0, alpha: float = 0.05,
                 n_permutations: int = 999):
        self.R = R
        self.theta = theta
        self.lambda_threshold = lambda_threshold
        self.alpha = alpha
        self.n_permutations = n_permutations

        # Internal state
        self.flows: List[Flow] = []
        self.neighborhoods: List[Set[int]] = []
        self.densities: List[float] = []
        self.g_star_values: List[float] = []
        self.seed_indices: Set[int] = set()
        self.clusters: List[Set[int]] = []
        self.significant_clusters: List[Set[int]] = []

    def fit(self, flows: List[Flow]) -> List[Set[int]]:
        """
        Execute the complete SDBC algorithm

        Parameters:
        -----------
        flows : List[Flow]
            List of flows to cluster

        Returns:
        --------
        List[Set[int]] : List of significant flow clusters (each as set of flow indices)
        """
        self.flows = flows
        n = len(flows)

        print(f"Starting SDBC clustering with {n} flows...")

        # Step 1: Build spatial neighborhoods and compute densities
        print("Step 1: Building spatial neighborhoods...")
        self._build_neighborhoods()

        print("Step 2: Computing flow densities...")
        self._compute_densities()

        # Step 2: Identify high-density flows using Getis-Ord G* statistic
        print("Step 3: Identifying high-density flows...")
        self._compute_g_star_statistic()
        self._identify_high_density_flows()

        print(f"  Found {len(self.seed_indices)} high-density seeds")

        # Step 3: Cluster with statistical constraint
        print("Step 4: Clustering with statistical constraint...")
        self._cluster_with_constraint()

        print(f"  Generated {len(self.clusters)} candidate clusters")

        # Step 4: Statistical significance test
        print("Step 5: Testing cluster significance...")
        self._test_cluster_significance()

        print(f"  Retained {len(self.significant_clusters)} significant clusters")

        return self.significant_clusters

    def _build_neighborhoods(self):
        """Build spatial neighborhoods based on spatial proximity, temporal similarity, and directional similarity"""
        n = len(self.flows)
        self.neighborhoods = [set() for _ in range(n)]

        # Build KD-trees for origins and destinations for efficient spatial queries
        origin_points = [f.origin for f in self.flows]
        dest_points = [f.destination for f in self.flows]

        origin_tree = KDTree(origin_points)
        dest_tree = KDTree(dest_points)

        for i, flow_i in enumerate(self.flows):
            # Find origins within buffer of flow_i's origin
            origin_neighbors = origin_tree.query_ball_point(flow_i.origin, self.R)

            for j in origin_neighbors:
                if i == j:
                    continue

                flow_j = self.flows[j]

                # Check spatial proximity: destination of flow_j within buffer of flow_i's destination
                dest_dist = np.linalg.norm(np.array(flow_j.destination) - np.array(flow_i.destination))

                if dest_dist <= self.R:
                    # Check directional similarity
                    angle = flow_i.angle_to(flow_j)

                    if angle <= self.theta:
                        # Check temporal similarity (if timestamps available)
                        temporal_ok = self._check_temporal_similarity(flow_i, flow_j)

                        if temporal_ok:
                            self.neighborhoods[i].add(j)
                            self.neighborhoods[j].add(i)

    def _check_temporal_similarity(self, flow_i: Flow, flow_j: Flow) -> bool:
        """Check if two flows have temporal overlap"""
        if flow_i.timestamp_origin is None or flow_i.timestamp_dest is None:
            return True  # No temporal constraint if timestamps not provided

        if flow_j.timestamp_origin is None or flow_j.timestamp_dest is None:
            return True

        # Check if time periods intersect
        period_i = (flow_i.timestamp_origin, flow_i.timestamp_dest)
        period_j = (flow_j.timestamp_origin, flow_j.timestamp_dest)

        # Intersection exists if start_i <= end_j and start_j <= end_i
        return period_i[0] <= period_j[1] and period_j[0] <= period_i[1]

    def _compute_densities(self):
        """Compute density for each flow as number of flows in its neighborhood"""
        self.densities = [len(neighbors) for neighbors in self.neighborhoods]

    def _compute_g_star_statistic(self):
        """Compute Getis-Ord G* statistic for each flow"""
        n = len(self.flows)
        z_mean = np.mean(self.densities)
        z_std = np.std(self.densities)

        if z_std == 0:
            self.g_star_values = [0.0] * n
            return

        self.g_star_values = []

        for i in range(n):
            # Sum of densities in neighborhood
            sum_neighbor_densities = self.densities[i]
            for j in self.neighborhoods[i]:
                sum_neighbor_densities += self.densities[j]

            # Number of neighbors (including self)
            w_sum = len(self.neighborhoods[i]) + 1

            # Calculate G* statistic
            numerator = sum_neighbor_densities - w_sum * z_mean
            denominator = z_std * np.sqrt((n * w_sum - w_sum ** 2) / (n - 1))

            if denominator == 0:
                g_star = 0.0
            else:
                g_star = numerator / denominator

            self.g_star_values.append(g_star)

    def _identify_high_density_flows(self):
        """Identify high-density flows (seeds) based on G* statistic threshold"""
        self.seed_indices = set()

        for i, g_star in enumerate(self.g_star_values):
            if g_star > self.lambda_threshold:
                self.seed_indices.add(i)

    def _cluster_with_constraint(self):
        """Perform statistical constrained clustering"""
        # Initialize clusters from seeds
        self.clusters = [{seed} for seed in self.seed_indices]
        cluster_neighborhoods = [self._get_cluster_neighborhood(cluster) for cluster in self.clusters]

        # Track merged clusters
        merged = [False] * len(self.clusters)

        changed = True
        iteration = 0
        max_iterations = len(self.clusters) * 2

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for i in range(len(self.clusters)):
                if merged[i]:
                    continue

                cluster_i = self.clusters[i]
                neighborhood_i = cluster_neighborhoods[i]

                if not neighborhood_i:
                    continue

                # Find best neighbor to merge
                best_j = -1
                best_g_star = -float('inf')

                for neighbor_idx in neighborhood_i:
                    # Find which cluster contains this neighbor
                    for j in range(len(self.clusters)):
                        if merged[j]:
                            continue
                        if neighbor_idx in self.clusters[j]:
                            # Compute G* for merged cluster
                            merged_cluster = cluster_i.union(self.clusters[j])
                            g_star_merged = self._compute_cluster_g_star(merged_cluster)

                            # Check if G* increases
                            current_g_star = self._compute_cluster_g_star(cluster_i)

                            if g_star_merged > current_g_star and g_star_merged > best_g_star:
                                best_g_star = g_star_merged
                                best_j = j
                            break

                # Merge if beneficial
                if best_j >= 0 and not merged[best_j]:
                    self.clusters[i] = self.clusters[i].union(self.clusters[best_j])
                    cluster_neighborhoods[i] = self._get_cluster_neighborhood(self.clusters[i])
                    merged[best_j] = True
                    changed = True

        # Filter out merged clusters
        self.clusters = [self.clusters[i] for i in range(len(self.clusters)) if not merged[i]]

        # Remove very small clusters (likely noise)
        min_cluster_size = 3
        self.clusters = [c for c in self.clusters if len(c) >= min_cluster_size]

    def _get_cluster_neighborhood(self, cluster: Set[int]) -> Set[int]:
        """Get the neighborhood of a cluster (all flows neighboring any flow in cluster, excluding cluster members)"""
        neighborhood = set()

        for flow_idx in cluster:
            neighborhood.update(self.neighborhoods[flow_idx])

        return neighborhood - cluster

    def _compute_cluster_g_star(self, cluster: Set[int]) -> float:
        """Compute G* statistic for a cluster"""
        n = len(self.flows)
        z_mean = np.mean(self.densities)
        z_std = np.std(self.densities)

        if z_std == 0:
            return 0.0

        # Sum densities of flows in cluster
        sum_densities = sum(self.densities[i] for i in cluster)

        w_c = len(cluster)

        numerator = sum_densities - w_c * z_mean
        denominator = z_std * np.sqrt((n * w_c - w_c ** 2) / (n - 1))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _test_cluster_significance(self):
        """Test statistical significance of candidate clusters using Monte Carlo permutation"""
        self.significant_clusters = []

        for cluster in self.clusters:
            p_value = self._compute_monte_carlo_p_value(cluster)

            if p_value < self.alpha:
                self.significant_clusters.append(cluster)
                print(f"  Cluster with {len(cluster)} flows: p-value = {p_value:.4f} (significant)")
            else:
                print(f"  Cluster with {len(cluster)} flows: p-value = {p_value:.4f} (not significant)")

    def _compute_monte_carlo_p_value(self, cluster: Set[int]) -> float:
        """Compute Monte Carlo p-value for a cluster"""
        # Compute domain range (convex hull + buffer)
        domain_range = self._get_cluster_domain_range(cluster)

        # Observed number of flows in domain range
        observed_count = self._count_flows_in_domain(cluster, domain_range)

        # Monte Carlo simulation
        n_flows = len(self.flows)
        simulated_counts = []

        for _ in range(self.n_permutations):
            # Random permutation: randomly pair origins and destinations
            origins = [self.flows[i].origin for i in range(n_flows)]
            destinations = [self.flows[i].destination for i in range(n_flows)]

            np.random.shuffle(destinations)

            # Count flows in domain range
            count = 0
            for i in range(n_flows):
                if self._is_flow_in_domain(origins[i], destinations[i], domain_range):
                    count += 1

            simulated_counts.append(count)

        # Compute p-value
        p_value = (sum(1 for c in simulated_counts if c >= observed_count) + 1) / (self.n_permutations + 1)

        return p_value

    def _get_cluster_domain_range(self, cluster: Set[int]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get domain range of a cluster (convex hull of origins and destinations with buffer)
        Returns: (origin_convex_hull, destination_convex_hull, buffer_size)
        """
        origins = [self.flows[i].origin for i in cluster]
        destinations = [self.flows[i].destination for i in cluster]

        origin_hull = self._compute_convex_hull(origins)
        dest_hull = self._compute_convex_hull(destinations)

        return (origin_hull, dest_hull, self.R)

    def _compute_convex_hull(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """Compute convex hull of points"""
        if len(points) < 3:
            return np.array(points)

        points_array = np.array(points)

        try:
            hull = ConvexHull(points_array)
            return points_array[hull.vertices]
        except Exception:
            return points_array

    def _count_flows_in_domain(self, cluster: Set[int], domain_range: Tuple) -> int:
        """Count number of flows (from the entire dataset) that fall within domain range"""
        origin_hull, dest_hull, buffer_size = domain_range

        count = 0
        for flow_idx, flow in enumerate(self.flows):
            if self._is_flow_in_domain(flow.origin, flow.destination, domain_range):
                count += 1

        return count

    def _is_flow_in_domain(self, origin: Tuple[float, float],
                           destination: Tuple[float, float],
                           domain_range: Tuple) -> bool:
        """Check if a flow is within the domain range"""
        origin_hull, dest_hull, buffer_size = domain_range

        # Simple check: if points are within buffered bounding box
        if len(origin_hull) > 0:
            origin_in = self._point_in_convex_hull(origin, origin_hull, buffer_size)
            dest_in = self._point_in_convex_hull(destination, dest_hull, buffer_size)
            return origin_in and dest_in

        return False

    def _point_in_convex_hull(self, point: Tuple[float, float],
                              hull_points: np.ndarray,
                              buffer_size: float) -> bool:
        """Check if a point is inside convex hull (with buffer)"""
        if len(hull_points) < 3:
            # For points with less than 3 points, use distance-based check
            if len(hull_points) == 1:
                dist = np.linalg.norm(np.array(point) - hull_points[0])
                return dist <= buffer_size
            elif len(hull_points) == 2:
                # Check distance to line segment
                p = np.array(point)
                a = hull_points[0]
                b = hull_points[1]

                # Project point onto line
                ab = b - a
                t = np.dot(p - a, ab) / np.dot(ab, ab)
                t = np.clip(t, 0, 1)
                closest = a + t * ab
                dist = np.linalg.norm(p - closest)
                return dist <= buffer_size

        # Use winding number algorithm with buffer expansion
        hull_points_2d = hull_points[:, :2] if hull_points.shape[1] > 2 else hull_points
        point_array = np.array(point)

        # First check if point is inside the convex hull
        inside = self._point_in_polygon(point_array, hull_points_2d)

        if inside:
            return True

        # Check distance to hull edges
        for i in range(len(hull_points_2d)):
            a = hull_points_2d[i]
            b = hull_points_2d[(i + 1) % len(hull_points_2d)]

            # Distance from point to line segment
            ab = b - a
            t = np.dot(point_array - a, ab) / np.dot(ab, ab)
            t = np.clip(t, 0, 1)
            closest = a + t * ab
            dist = np.linalg.norm(point_array - closest)

            if dist <= buffer_size:
                return True

        return False

    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Ray casting algorithm to check if point is inside polygon"""
        x, y = point
        inside = False
        n = len(polygon)

        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            # Check if point is on the edge
            if self._point_on_segment(point, (x1, y1), (x2, y2)):
                return True

            # Check if ray crosses the edge
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside

        return inside

    def _point_on_segment(self, point: np.ndarray,
                          seg_start: Tuple[float, float],
                          seg_end: Tuple[float, float]) -> bool:
        """Check if point lies on line segment"""
        p = point
        a = np.array(seg_start)
        b = np.array(seg_end)

        # Check collinearity
        cross = np.cross(b - a, p - a)
        if abs(cross) > 1e-10:
            return False

        # Check if within bounding box
        dot = np.dot(p - a, b - a)
        if dot < 0 or dot > np.dot(b - a, b - a):
            return False

        return True


class RDVParameterEstimator:
    """
    Ratio of spatial proximity Density Variance (RDV) for adaptive parameter selection
    """

    @staticmethod
    def estimate_optimal_parameters(flows: List[Flow],
                                    R_range: Tuple[float, float] = (150, 500),
                                    theta_range: Tuple[float, float] = (0, 45),
                                    R_step: float = 10,
                                    theta_step: float = 5) -> Tuple[float, float]:
        """
        Estimate optimal R and theta parameters using RDV

        Parameters:
        -----------
        flows : List[Flow]
            List of flows
        R_range : tuple
            (min_R, max_R) in meters
        theta_range : tuple
            (min_theta, max_theta) in degrees
        R_step : float
            Step size for R
        theta_step : float
            Step size for theta

        Returns:
        --------
        Tuple[float, float] : Optimal (R, theta)
        """
        R_vals = np.arange(R_range[0], R_range[1] + R_step, R_step)
        theta_vals = np.arange(theta_range[0], theta_range[1] + theta_step, theta_step)

        rdv_matrix = np.zeros((len(R_vals), len(theta_vals)))

        for i, R in enumerate(R_vals):
            for j, theta in enumerate(theta_vals):
                # Compute density variance for current parameters
                var_current = RDVParameterEstimator._compute_density_variance(flows, R, theta)

                # Compute density variance for incremented parameters
                R_next = min(R + R_step, R_range[1])
                theta_next = min(theta + theta_step, theta_range[1])
                var_next = RDVParameterEstimator._compute_density_variance(flows, R_next, theta_next)

                # RDV calculation
                if var_current > 0:
                    rdv_matrix[i, j] = (var_next / var_current) / (R_next / R)
                else:
                    rdv_matrix[i, j] = float('inf')

        # Find parameters with minimum RDV
        min_idx = np.unravel_index(np.argmin(rdv_matrix), rdv_matrix.shape)
        optimal_R = R_vals[min_idx[0]]
        optimal_theta = theta_vals[min_idx[1]]

        return optimal_R, optimal_theta

    @staticmethod
    def _compute_density_variance(flows: List[Flow], R: float, theta: float) -> float:
        """Compute spatial neighborhood density variance for given parameters"""
        n = len(flows)

        # Build neighborhoods with given parameters
        neighborhoods = []

        origin_points = [f.origin for f in flows]
        dest_points = [f.destination for f in flows]

        origin_tree = KDTree(origin_points)
        dest_tree = KDTree(dest_points)

        for i, flow_i in enumerate(flows):
            neighbors = set()
            origin_neighbors = origin_tree.query_ball_point(flow_i.origin, R)

            for j in origin_neighbors:
                if i == j:
                    continue

                flow_j = flows[j]
                dest_dist = np.linalg.norm(np.array(flow_j.destination) - np.array(flow_i.destination))

                if dest_dist <= R:
                    angle = flow_i.angle_to(flow_j)
                    if angle <= theta:
                        neighbors.add(j)

            neighborhoods.append(neighbors)

        densities = [len(neighbors) for neighbors in neighborhoods]

        return np.var(densities) if len(densities) > 1 else 0


# Utility functions for evaluation
def calculate_ari(true_clusters: List[Set[int]], pred_clusters: List[Set[int]], n_flows: int) -> float:
    """
    Calculate Adjusted Rand Index (ARI) for clustering evaluation

    Parameters:
    -----------
    true_clusters : List[Set[int]]
        Ground truth clusters (each as set of flow indices)
    pred_clusters : List[Set[int]]
        Predicted clusters
    n_flows : int
        Total number of flows
    """
    # Build contingency table
    contingency = np.zeros((len(true_clusters) + 1, len(pred_clusters) + 1), dtype=int)

    # Map each flow to its true and predicted cluster
    true_membership = {}
    for i, cluster in enumerate(true_clusters):
        for flow_idx in cluster:
            true_membership[flow_idx] = i

    pred_membership = {}
    for j, cluster in enumerate(pred_clusters):
        for flow_idx in cluster:
            pred_membership[flow_idx] = j

    # Fill contingency table
    for flow_idx in range(n_flows):
        true_i = true_membership.get(flow_idx, len(true_clusters))
        pred_j = pred_membership.get(flow_idx, len(pred_clusters))
        contingency[true_i, pred_j] += 1

    # Calculate ARI
    n = n_flows

    sum_combos = 0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            n_ij = contingency[i, j]
            if n_ij >= 2:
                sum_combos += n_ij * (n_ij - 1) / 2

    sum_i = 0
    for i in range(contingency.shape[0]):
        n_i = np.sum(contingency[i, :])
        if n_i >= 2:
            sum_i += n_i * (n_i - 1) / 2

    sum_j = 0
    for j in range(contingency.shape[1]):
        n_j = np.sum(contingency[:, j])
        if n_j >= 2:
            sum_j += n_j * (n_j - 1) / 2

    expected_index = sum_i * sum_j / (n * (n - 1) / 2)
    max_index = (sum_i + sum_j) / 2

    if max_index == expected_index:
        return 1.0

    ari = (sum_combos - expected_index) / (max_index - expected_index)

    return max(0, min(1, ari))


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic test data
    np.random.seed(42)


    def generate_cluster_flows(center_origin: Tuple[float, float],
                               center_dest: Tuple[float, float],
                               n_flows: int,
                               spread: float = 50.0) -> List[Flow]:
        """Generate a cluster of flows around given centers"""
        flows = []
        for i in range(n_flows):
            origin = (center_origin[0] + np.random.randn() * spread,
                      center_origin[1] + np.random.randn() * spread)
            dest = (center_dest[0] + np.random.randn() * spread,
                    center_dest[1] + np.random.randn() * spread)
            flows.append(Flow(flow_id=i, origin=origin, destination=dest))
        return flows


    def generate_noise_flows(n_flows: int, bounds: Tuple[float, float, float, float]) -> List[Flow]:
        """Generate random noise flows"""
        flows = []
        start_id = len(flows)
        for i in range(n_flows):
            origin = (np.random.uniform(bounds[0], bounds[1]),
                      np.random.uniform(bounds[2], bounds[3]))
            dest = (np.random.uniform(bounds[0], bounds[1]),
                    np.random.uniform(bounds[2], bounds[3]))
            flows.append(Flow(flow_id=start_id + i, origin=origin, destination=dest))
        return flows


    # Create test data with three clusters
    print("Creating synthetic test data...")

    clusters = [
        ((100, 100), (200, 200), 50),  # Cluster 1: 50 flows
        ((300, 300), (400, 400), 50),  # Cluster 2: 50 flows
        ((150, 500), (250, 600), 50),  # Cluster 3: 50 flows
    ]

    all_flows = []
    true_clusters = []
    flow_id = 0

    for cluster_idx, (origin_center, dest_center, size) in enumerate(clusters):
        cluster_flows = []
        for _ in range(size):
            origin = (origin_center[0] + np.random.randn() * 30,
                      origin_center[1] + np.random.randn() * 30)
            dest = (dest_center[0] + np.random.randn() * 30,
                    dest_center[1] + np.random.randn() * 30)
            flow = Flow(flow_id=flow_id, origin=origin, destination=dest)
            all_flows.append(flow)
            cluster_flows.append(flow_id)
            flow_id += 1
        true_clusters.append(set(cluster_flows))

    flow_shp = read_shapefile(rf'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\datasets\aggreation patterns\FD7.shp')
    flowReader = FlowReader()
    all_flows, true_clusters = flowReader.shp_to_flows(flow_df=flow_shp, origin_cols=('ox', 'oy'), dest_cols=('dx', 'dy'), id_col='oid')
    print(all_flows)

    # Add noise flows
    # n_noise = 50
    # bounds = (0, 700, 0, 700)
    # for _ in range(n_noise):
    #     origin = (np.random.uniform(bounds[0], bounds[1]),
    #               np.random.uniform(bounds[2], bounds[3]))
    #     dest = (np.random.uniform(bounds[0], bounds[1]),
    #             np.random.uniform(bounds[2], bounds[3]))
    #     flow = Flow(flow_id=flow_id, origin=origin, destination=dest)
    #     all_flows.append(flow)
    #     flow_id += 1

    print(f"Total flows: {len(all_flows)}")
    print(f"True clusters: {[len(c) for c in true_clusters]}")

    # Run SDBC clustering
    print("\n" + "=" * 60)
    print("Running SDBC algorithm...")
    print("=" * 60)

    sdbc = SDBC(R=400.0, theta=30.0, lambda_threshold=0, alpha=0.05, n_permutations=99)
    detected_clusters = sdbc.fit(all_flows)
    print(detected_clusters)
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Detected clusters: {len(detected_clusters)}")
    for i, cluster in enumerate(detected_clusters):
        print(f"  Cluster {i + 1}: {len(cluster)} flows")

    # Evaluate with ARI
    if len(detected_clusters) > 0:
        ari = calculate_ari(true_clusters, detected_clusters, len(all_flows))
        print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")

    # Parameter estimation example
    print("\n" + "=" * 60)
    print("Parameter estimation with RDV...")
    print("=" * 60)

    # For demonstration, use a subset of data
    # sample_flows = all_flows[:200]
    #
    # try:
    #     optimal_R, optimal_theta = RDVParameterEstimator.estimate_optimal_parameters(
    #         sample_flows,
    #         R_range=(80, 150),
    #         theta_range=(10, 40),
    #         R_step=10,
    #         theta_step=5
    #     )
    #     print(f"Estimated optimal parameters: R = {optimal_R:.1f}, theta = {optimal_theta:.1f}")
    # except Exception as e:
    #     print(f"Parameter estimation failed: {e}")
    #     print("Using default parameters instead")
