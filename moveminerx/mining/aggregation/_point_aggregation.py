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
from moveminerx.mining.aggregation._trajectory_aggregation import evaluate_predicted_clusters, \
    saved_quantitative_results
from moveminerx.utils.util import load_mat, read_shapefile, save_shapefile, save_json, load_json
# from cluster_result_evaluation import baseline_results_evaluation, get_cluster_labels_pred, \
#     Adjusted_Rand_index, NMI_index
from tests.evaluations import Adjusted_Rand_index, get_cluster_labels_pred


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


def dbscan_clustering(dist, X=None, eps=6, min_samples=2, metric='euclidean'):
    """"""
    # I5: eps=1.30
    # T1, T2, T3 = 5, 10, 10
    # cross = 10
    # I5SIM: eps=10
    if metric == 'euclidean':
        dbscan_res = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(X)
    else:
        dbscan_res = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(dist)
    # print(np.unique(dbscan_res.labels_))
    return dbscan_res


def K_mediods_clustering(dist, n_clusters=4):
    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
    res = kmedoids.fit(dist)
    return res


def mean_shift_clustering(dist, bandwidth=None):
    bandwidth = estimate_bandwidth(dist, quantile=0.2, n_samples=len(dist))
    res = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(dist)
    return res


def K_means_clustering(dist, X=None, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", algorithm="auto")
    if X is not None:
        kmeans.fit(X)
    else:
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
    spectral clustering
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



def main():
    root_folder = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\datasets\aggreation patterns'
    saved_folder = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\results'
    datasets = ['PD1']
    real_n_clusters = {'PD1': 9, }
    methods = ['dbscan', 'k-means', 'hdbscan', 'affinity_propagation', 'spectral_clustering', 'DPC', 'hierarchical_clustering', 'k-medoids']
    start = 0
    end = 1
    for method in methods[1:2]:
        dist = np.ones_like((1, 1))
        if method == 'hdbscan':
            for dataset in datasets[start: end]:
                shp_path = r'{}\{}\{}.shp'.format(root_folder, dataset, dataset)
                data_shp = read_shapefile(shp_path)
                min_cluster_size = 4
                t1 = time.time()
                res = hdbscan_clustering(dist, min_cluster_size=min_cluster_size)
                t2 = time.time()
                t = t2 * 1000 - t1 * 1000
                label = res.labels_
        elif method == 'dbscan':
            for dataset in datasets[start: end]:
                shp_path = rf'{root_folder}\{dataset}.shp'
                data_shp = read_shapefile(shp_path)
                best_eps = 0
                best_ARI = 0
                best_min_samples = 0
                min_samples_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                geos = data_shp['geometry'].values.tolist()
                X = np.array([geo.coords[0] for geo in geos])
                for min_samples in min_samples_list:
                    for i in range(5, 20, 1):
                        eps = i
                        # eps = i / 2
                        try:
                            res = dbscan_clustering(dist, X=X, eps=eps, min_samples=min_samples)
                        except:
                            continue

                        data_shp['new_IDX'] = res.labels_
                        clusters = evaluate_predicted_clusters(data_shp)
                        labels_pred = get_cluster_labels_pred(clusters)
                        labels = data_shp.IDX.values.tolist()
                        ARI = Adjusted_Rand_index(labels_pred, labels)
                        if ARI > best_ARI:
                            best_ARI = ARI
                            best_eps = eps
                            best_min_samples = min_samples
                eps = best_eps
                print(eps, best_min_samples)
                res = dbscan_clustering(dist, X=X, eps=eps, min_samples=best_min_samples)
                data_shp['new_IDX'] = res.labels_
                clusters = evaluate_predicted_clusters(data_shp)
                labels_pred = get_cluster_labels_pred(clusters)
                print(set(labels_pred))
                labels = data_shp.IDX.values.tolist()
                ARI = Adjusted_Rand_index(labels_pred, labels)

                print(f'DBSCAN: best_eps: {eps}, min_samples: {best_min_samples}, ARI: {ARI}')

                if not os.path.exists(rf'{saved_folder}\shps\{method}'):
                    os.mkdir(rf'{saved_folder}\shps\{method}')
                saved_shp_path = rf'{saved_folder}\shps\{method}\{dataset}_{method}_eps{eps}_ms{best_min_samples}.shp'
                save_shapefile(saved_shp_path, data_shp)
                saved_quantitative_results(dataset=dataset, clusters=clusters, labels_real=labels,
                                           labels_pred=labels_pred, dist_matrix=dist,
                                           real_num_clusters=real_n_clusters[dataset], saved_shp_path=saved_shp_path,
                                           saved_folder=saved_folder, method=method,
                                           params={"eps": eps, "min_samples": best_min_samples})

        elif method == 'k-means':
            for dataset in datasets[start:end]:
                shp_path = rf'{root_folder}\{dataset}.shp'
                data_shp = read_shapefile(shp_path)
                geos = data_shp['geometry'].values.tolist()
                X = np.array([geo.coords[0] for geo in geos])
                t1 = time.time()
                res = K_means_clustering(dist, X=X, n_clusters=real_n_clusters[dataset])
                data_shp['new_IDX'] = res.labels_
                clusters = evaluate_predicted_clusters(data_shp)
                labels_pred = get_cluster_labels_pred(clusters)
                t2 = time.time()
                t = t2 * 1000 - t1 * 1000
                label = res.labels_
                if not os.path.exists(rf'{saved_folder}\shps\{method}'):
                    os.mkdir(rf'{saved_folder}\shps\{method}')
                saved_shp_path = rf'{saved_folder}\shps\{method}\{dataset}_{method}_n_clusters{real_n_clusters[dataset]}.shp'
                labels = data_shp.IDX.values.tolist()
                save_shapefile(saved_shp_path, data_shp)
                saved_quantitative_results(dataset=dataset, clusters=clusters, labels_real=labels,
                                           labels_pred=labels_pred, dist_matrix=dist,
                                           real_num_clusters=real_n_clusters[dataset], saved_shp_path=saved_shp_path,
                                           saved_folder=saved_folder, method=method,
                                           params={"n_clusters": real_n_clusters[dataset]})

        elif method == 'hierarchical_clustering':
            linkages = ['single', 'complete', 'average', 'ward']
            for linkage in linkages:
                if linkage == 'ward':
                    affinity = 'euclidean'
                else:
                    affinity = 'precomputed'
                for dataset in datasets[start: end]:
                    shp_path = r'{}\{}\{}.shp'.format(root_folder, dataset, dataset)
                    data_shp = read_shapefile(shp_path)
                    t1 = time.time()
                    res = hierarchical_clustering(dist, linkage=linkage, n_clusters=real_n_clusters[dataset],
                                                  affinity=affinity)
                    t2 = time.time()
                    t = t2 * 1000 - t1 * 1000
                    label = res.labels_




        elif method == 'mean-shift':
            for dataset in datasets[start:end]:

                # shp_path = r'{}\simulated_datasets\{}\{}.shp'.format(root_folder, dataset, dataset)
                data_shp = read_shapefile(shp_path)
                res = mean_shift_clustering(dist)

        elif method == 'spectral_clustering':
            for dataset in datasets[start:end]:
                shp_path = r'{}\{}\{}.shp'.format(root_folder, dataset, dataset)
                data_shp = read_shapefile(shp_path)
                t1 = time.time()
                res = spectral_clustering(dist, n_clusters=real_n_clusters[dataset])
                t2 = time.time()
                t = t2 * 1000 - t1 * 1000
                label = res.labels_

        elif method == 'affinity_propagation':
            for dataset in datasets[start:end]:

                shp_path = r'{}\{}\{}.shp'.format(root_folder, dataset, dataset)
                data_shp = read_shapefile(shp_path)
                t1 = time.time()
                res = affinity_clustering(dist)
                t2 = time.time()
                t = t2 * 1000 - t1 * 1000
                label = res.labels_

        elif method == 'DPC':
            for dataset in datasets[start: end]:
                shp_path = r'{}\{}\{}.shp'.format(root_folder, dataset, dataset)
                data_shp = read_shapefile(shp_path)
                t1 = time.time()
                n_clusters_list = [9, 6, 9, 8, 10, 15, 6]
                n_clusters_list = [10, 16, 9, 8, 12]  # dspd
                n_clusters_list = [7, 19, 10, 5, 11]  # hausdorff
                n_clusters_list = [7, 20, 8, 5, 12]  # dtw
                # n_clusters = n_clusters_list[datasets.index(dataset)]
                estimated_num_clusters = n_clusters_list[datasets.index(dataset)]
                labels = DPC_clusters(dist, n_clusters=estimated_num_clusters, dataset=dataset)
                t2 = time.time()
                t = t2 * 1000 - t1 * 1000

        elif method == 'k-medoids':
            for dataset in datasets[start:end]:
                shp_path = r'{}\{}\{}.shp'.format(root_folder, dataset, dataset)
                data_shp = read_shapefile(shp_path)
                res = K_mediods_clustering(dist, n_clusters=real_n_clusters[dataset])
                label = res.labels_



if __name__ == "__main__":
    main()
