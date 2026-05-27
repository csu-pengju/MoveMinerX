#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: _point_aggregation.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/4/22 10:56
import os
import time
from typing import Dict

import hdbscan
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering, AffinityPropagation, KMeans, MeanShift, \
    estimate_bandwidth
from sklearn.cluster import OPTICS, Birch
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn_extra.cluster import KMedoids
from DPC import *
# from Traj_Cluster2.new_cluster_result_evaluation import cluster_result_evaluation, save_evaluation_results
from moveminerx.utils.util import load_mat, read_shapefile, save_shapefile, save_json, load_json
# from cluster_result_evaluation import baseline_results_evaluation, get_cluster_labels_pred, \
#     Adjusted_Rand_index, NMI_index
from tests.evaluations import Adjusted_Rand_index, get_cluster_labels_pred, cluster_result_evaluation


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


def hdbscan_clustering(dist, min_cluster_size=2, metric='precomputed'):
    if metric == 'precomputed':
        h_res = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, gen_min_span_tree=True).fit(dist)
    else:
        h_res = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, gen_min_span_tree=True).fit(dist)
    return h_res


def dbscan_clustering(dist, eps=6, min_samples=2):
    # I5: eps=1.30
    # T1, T2, T3 = 5, 10, 10
    # cross = 10
    # I5SIM: eps=10
    dbscan_res = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dist)
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


#
# def saved_baseline_results(dataset=None, clusters=None, labels_pred=None, labels_real=None,
#                            dist_matrix=None, real_num_clusters=None, saved_shp_path=None, saved_folder=None, method=None):
#     [SI_real, SI, ARI, Precision, Recall, NMI, AMI, entropy, Dunn, purity, ch_score, DBI, homogeneity,
#      completeness, v_measure_score] = cluster_result_evaluation(clusters, labels_real, dist_matrix)
#     save_evaluation_results(dataset=dataset, SI_real=SI_real, SI=SI, ARI=ARI, Precision=Precision,
#                             Recall=Recall,
#                             NMI=NMI, AMI=AMI, entropy=entropy, Dunn=Dunn, purity=purity,
#                             ch_score=ch_score, DBI=DBI,
#                             homogeneity=homogeneity, completeness=completeness,
#                             v_measure_score=v_measure_score,
#                             n_clusters=len(set(list(labels_pred))), real_clusters=real_num_clusters,
#                             root_folder=saved_folder, method=method, r_type='clustering',
#                             filepath=saved_shp_path)


def evaluate_predicted_clusters(df: pd.DataFrame):
    groups = df.groupby(by=['new_IDX'])
    clusters = {}
    for group in groups:
        IDX = group[0]
        try:
            Ids = group[1].Id.values.tolist()
        except:
            Ids = group[1].id.values.tolist()
        clusters[IDX] = Ids

    return clusters


def saved_quantitative_results(dataset=None, clusters=None, labels_pred=None, labels_real=None,
                               dist_matrix=None, real_num_clusters=None, saved_shp_path=None, saved_folder=None,
                               method=None, params: Dict = None):
    [SI, ARI, Precision, Recall, F1, NMI, AMI, entropy, Dunn, purity, ch_score, DBI, homogeneity,
     completeness, v_measure_score] = cluster_result_evaluation(clusters, labels_real, dist_matrix)
    save_evaluation_results(dataset=dataset, SI=SI, ARI=ARI, Precision=Precision,
                            Recall=Recall,
                            NMI=NMI, AMI=AMI, entropy=entropy, Dunn=Dunn, purity=purity,
                            ch_score=ch_score, DBI=DBI, F1_score=F1,
                            homogeneity=homogeneity, completeness=completeness,
                            v_measure_score=v_measure_score,
                            n_clusters=len(set(list(labels_pred))), real_clusters=real_num_clusters,
                            root_folder=saved_folder, method=method, r_type='clustering',
                            filepath=saved_shp_path, params=params)


def save_evaluation_results(SI=1.0, ARI=0.0, Recall=0.0, Precision=0.0, NMI=0.0, AMI=0.0, DBI=0.0, Dunn=0.0,
                            entropy=0.0, purity=0.0, dataset='T1', DBI2=0.0, ch_score=0.0, homogeneity=0.0,
                            completeness=0.0, F1_score=0.0,
                            n_clusters=1, real_clusters=1, r_type='clustering_results', filepath='',
                            method='proposed', root_folder='', t=0, v_measure_score=1.0, params: Dict = None):
    """
    保存簇的评价指标结果
    """

    metrics_json = {'dataset': dataset, 'r_type': r_type, 'method': method, 'SI': SI, 'ARI': ARI, 'NMI': NMI,
                    'AMI': AMI, 'Precision': Precision, 'Recall': Recall, 'F1_score': F1_score, 'DBI': DBI,
                    'DBI2': DBI2, 'Dunn': Dunn, 'ch_score': ch_score, 'homogeneity': homogeneity,
                    'completeness': completeness, 'entropy': entropy, 'purity': purity, 'v_measure_score': v_measure_score,
                    'n_clusters': n_clusters, 'real_clusters': real_clusters, 'filepath': filepath, 'time': t,
                    'params': str(params)}

    # for param_key, param_value in params.items():
    #     metrics_json[param_key] = param_value
    # print(metrics_json)
    df = pd.json_normalize(metrics_json)
    if os.path.exists(rf'{root_folder}\clustering_results_evaluation.csv'):
        df.to_csv(rf'{root_folder}\clustering_results_evaluation.csv', header=False, index=False, mode='a+')
    else:
        df.to_csv(rf'{root_folder}\clustering_results_evaluation.csv', index=False)


def main():
    root_folder = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\datasets\aggreation patterns'
    saved_folder = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\results'

    real_n_clusters = {'ST9': 7, 'ST6': 8, 'ST2': 6, 'ST1': 10, 'ST3': 7, 'TC1': 8, 'TC2': 5, 'TC3': 9,
                       'TC4': 10, 'i5': 10, 'i5C': 11, 'i5sim': 8,
                       'i5simC': 12, 'cross': 23, 'ST4': 7, 'TD1': 10}
    datasets = ['TD1']
    methods = ['dbscan', 'hdbscan', 'affinity_propagation', 'spectral_clustering', 'DPC', 'hierarchical_clustering',
               'k-means', 'k-medoids']
    start = 0
    end = 1
    # metric = 'hausdorff'
    metric = 'dspd'
    # metric = 'hausdorff'
    for method in methods[:1]:
        dataset = 'TD1'
        if metric == 'dspd':
            dist = load_mat(f'{root_folder}\\distance matrix\\{dataset}_dspd.mat')['dspd']
        elif metric == 'hausdorff':
            dist = load_mat(f'{root_folder}\\distance matrix\\{dataset}_hausdorff.mat')['hausdorff']

        if method == 'hdbscan':
            for dataset in datasets[start: end]:
                # cross: 2-4, ST1: 5, ST2: 2, ST3: 3
                # hausdorff, ST3, 7
                # dtw: cross,
                shp_path = r'{}\{}\{}.shp'.format(root_folder, dataset, dataset)
                data_shp = read_shapefile(shp_path)
                min_cluster_size = 4
                t1 = time.time()
                res = hdbscan_clustering(dist, min_cluster_size=min_cluster_size)
                t = time.time() * 1000 - t1 * 1000
                label = res.labels_
                data_shp['new_IDX'] = label
                clusters = evaluate_predicted_clusters(data_shp)
                labels_pred = get_cluster_labels_pred(clusters)

                labels = data_shp.IDX.values.tolist()
                ARI = Adjusted_Rand_index(clusters, labels)
                # NMI, AMI = NMI_index(clusters, labels)
                SI = silhouette_score(dist, labels_pred, metric='precomputed')
                print(f'SI: {SI}, ARI: {ARI}')
                if not os.path.exists(r'{}\baselines\{}'.format(saved_folder, method)):
                    os.mkdir(r'{}\baselines\{}'.format(saved_folder, method))
                saved_shp_path = r'{}\baselines\{}\{}_{}_{}.shp'.format(saved_folder, method, dataset, method,
                                                                        min_cluster_size)
                # save_shapefile(saved_shp_path, data_shp)

        elif method == 'dbscan':
            for dataset in datasets[0: 1]:
                # dataset = 'TD1'
                shp_path = r'{}\{}_pro.shp'.format(root_folder, dataset)
                data_shp = read_shapefile(shp_path)

                min_samples_list = [2]
                best_eps = 0
                best_ARI = 0
                best_min_samples = 0
                for min_samples in min_samples_list:
                    # min_samples = 2
                    for i in range(2, 60, 1):
                        # eps = i*5
                        eps = i / 2
                        # eps = 4.5
                        try:
                            res = dbscan_clustering(dist, eps=eps, min_samples=min_samples)
                        except:
                            continue

                        data_shp['new_IDX'] = res.labels_
                        clusters = evaluate_predicted_clusters(data_shp)
                        labels_pred = get_cluster_labels_pred(clusters)
                        # SI = silhouette_score(dist, labels_pred, metric='precomputed')
                        labels = data_shp.IDX.values.tolist()
                        ARI = Adjusted_Rand_index(labels_pred, labels)
                        if ARI > best_ARI:
                            best_ARI = ARI
                            best_eps = eps
                            best_min_samples = min_samples
                        # NMI, AMI = NMI_index(clusters, labels)
                    # print(f'DBSCAN: eps: {eps}, SI: {SI}, ARI: {ARI}, AMI: {AMI}')
                # save_json(r'{}\{}_{}.json'.format(bl_json_folder, method, dataset), clusters)
                eps = best_eps
                res = dbscan_clustering(dist, eps=eps, min_samples=best_min_samples)
                data_shp['new_IDX'] = res.labels_
                clusters = evaluate_predicted_clusters(data_shp)
                labels_pred = get_cluster_labels_pred(clusters)
                print(set(labels_pred))
                labels = data_shp.IDX.values.tolist()
                ARI = Adjusted_Rand_index(labels_pred, labels)
                # NMI, AMI = NMI_index(clusters, labels)
                # SI = silhouette_score(dist, labels_pred, metric='precomputed')
                print(f'DBSCAN: best_eps: {eps}, min_samples: {best_min_samples}, ARI: {ARI}')

                if not os.path.exists(rf'{saved_folder}\shps\{method}'):
                    os.mkdir(rf'{saved_folder}\shps\{method}')
                saved_shp_path = rf'{saved_folder}\shps\{method}\{dataset}_{method}_{metric}_eps{eps}_ms{best_min_samples}.shp'
                save_shapefile(saved_shp_path, data_shp)
                saved_quantitative_results(dataset=dataset, clusters=clusters, labels_real=labels,
                                           labels_pred=labels_pred, dist_matrix=dist,
                                           real_num_clusters=real_n_clusters[dataset], saved_shp_path=saved_shp_path,
                                           saved_folder=saved_folder, method=method,
                                           params={"metric": metric, "eps": eps, "min_samples": best_min_samples})


if __name__ == "__main__":
    main()
