#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _point_aggregation.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/4/22 10:56
import os
import time

import hdbscan
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering, AffinityPropagation, KMeans, MeanShift, \
    estimate_bandwidth
from sklearn.cluster import OPTICS, Birch
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn_extra.cluster import KMedoids
from DPC import *
# from Traj_Cluster2.new_cluster_result_evaluation import cluster_result_evaluation, save_evaluation_results
from moveminerx.utils.util import load_mat, read_shapefile, save_shapefile, save_json, load_json
# from cluster_result_evaluation import baseline_results_evaluation, get_cluster_labels_pred, \
#     Adjusted_Rand_index, NMI_index


def hierarchical_clustering(dist, n_clusters=7, linkage='ward', affinity='precomputed'):
    """
    # 基于ward, average, complete, single的自下而上层次聚类
    :param dist:
    :param n_clusters:
    :param linkage:
    :param affinity:
    :return:
    """
    hi_res = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity).fit(dist)
    return hi_res


def hdbscan_clustering(dist, min_cluster_size=2):
    h_res = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed', gen_min_span_tree=True).fit(dist)
    return h_res


def dbscan_clustering(dist, eps=6, min_samples=2):
    # I5: eps=1.30
    # T1, T2, T3 = 5, 10, 10
    # cross = 10
    # I5SIM: eps=10
    dbscan_res = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dist)
    print(np.unique(dbscan_res.labels_))
    return dbscan_res


def K_mediods_clustering(dist, n_clusters=4):
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
    res = kmedoids.fit(dist)
    return res


def mean_shift_clustering(dist, bandwidth=None):
    bandwidth = estimate_bandwidth(dist, quantile=0.2, n_samples=len(dist))
    res = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(dist)
    return res


def K_means_clustering(dist, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", algorithm="auto")
    kmeans.fit(dist)
    # res = kmeans.predict(dist)
    return kmeans


def DPC_clusters(dist, n_clusters=8, dataset=None):
    dc = select_dc(dist)
    rhos = get_local_density(dist, dc)
    deltas, nearest_neighbor = get_deltas(dist, rhos)
    # plt.ylim(0, 120)
    plt.scatter(rhos, deltas, s=0.5, color='b')
    # plt.savefig(r'D:\Projects\TrajCluster\202407_TrajCluster\simulated_Ex\results\baselines\{}.jpg'.format(dataset),
    #             dpi=600)
    plt.show()
    centers = find_k_centers(rhos, deltas, n_clusters)
    labels = density_peal_cluster(rhos, centers, nearest_neighbor)
    return labels


def spectral_clustering(dist, n_clusters=2, affinity='precomputed'):
    """

    :param dist:
    :param n_clusters:
    :param affinity: nearest_neighbors or rbf or...
    :return:
    """
    res = SpectralClustering(n_clusters=n_clusters, affinity=affinity).fit(dist)
    return res


def affinity_clustering(dist):
    res = AffinityPropagation(affinity='precomputed').fit(dist)
    return res


def evaluate_real_data(df: pd.DataFrame):
    groups = df.groupby(by=['new_IDX'])
    clusters = {}
    for group in groups:
        IDX = group[0]
        Ids = group[1].Id.values.tolist()
        clusters[IDX] = Ids

    return clusters



if __name__ == "__main__":
    pass
