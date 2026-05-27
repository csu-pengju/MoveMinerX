#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: evaluations.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/4/21 21:53
import os
from math import inf
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, pair_confusion_matrix, homogeneity_completeness_v_measure, \
    calinski_harabasz_score, silhouette_score, davies_bouldin_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score

from moveminerx.mining._basic_class import MovingObject, ConvergingTreeNode, SnapshotBuilder, REMOConvergence, \
    ConvergingPattern
from tests.util import read_shapefile, save_shapefile


def custom_sort_key(key):
    return (key[0], key[1])


def match_pattern_truth(predicted: Dict[Tuple[int, str], int], truth: Dict[Tuple[int, str], int]) -> Dict[str, float]:
    """对比挖掘结果与真实标签的 (object_id, timestamp) 对，计算 Jaccard / F1 / ARI 等"""
    # all_items = list(set(predicted) | set(truth))
    sorted_predicted_keys = sorted(predicted.keys(), key=custom_sort_key)
    sorted_predicted_values = [predicted[key] for key in sorted_predicted_keys]
    sorted_truth_keys = sorted(truth.keys(), key=custom_sort_key)
    # print(sorted_truth_keys)
    # print(sorted_truth_keys)
    sorted_truth_values = [truth[key] for key in sorted_truth_keys]
    # y_true = [1 if item in truth else 0 for item in all_items]
    # y_pred = [1 if item in predicted else 0 for item in all_items]
    y_true = sorted_truth_values
    y_pred = sorted_predicted_values
    # print(f'y_true: {y_true}')
    # print(f'y_pred: {y_pred}')
    # jacc = jaccard_score(y_true, y_pred)
    ARI = adjusted_rand_score(y_true, y_pred)
    (tn, fp), (fn, tp) = pair_confusion_matrix(y_pred, y_true)

    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    # f1_micro = f1_score(y_true, y_pred, average='micro')
    # f1_macro = f1_score(y_true, y_pred, average='macro')
    # f1_macro2 = f1_score(y_true, y_pred, average=None)
    # print(f1_macro2)
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    return {
        'jaccard': 0,
        'f1': round(f1, 4),
        'ARI': round(ARI, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        # 'f1_micro': round(f1_micro, 4),
        # 'f1_macro': round(f1_macro, 4)
    }


def format_predicted_truth(patterns, objects: Dict[str, MovingObject], pattern_type='None') -> Dict[str, Dict]:
    predicted: Dict[Tuple[int, str], int] = {}
    truth: Dict[Tuple[int, str], int] = {}

    for oid, obj in objects.items():
        points = obj.trajectory.points
        for t, pt in points.items():
            truth[(t, oid)] = pt.truth
            predicted[(t, oid)] = -1

    if isinstance(patterns, dict):
        if pattern_type == 'convoy':
            for pid, pattern in patterns.items():
                timestamps = pattern.time_to_objects.keys()
                objects = pattern.objects
                for t in timestamps:
                    for oid in objects:
                        predicted[(t, oid)] = pid
        else:
            for pid, pattern in patterns.items():
                t_objs = pattern.time_to_objects
                for t, oids in t_objs.items():
                    for oid in oids:
                        if (t, oid) in predicted:
                            predicted[(t, oid)] = pid

    elif isinstance(patterns, list):
        if pattern_type == 'loose traveling companion':
            for pid, pattern in enumerate(patterns):
                timestamps = pattern.timestamps
                objects = pattern.object_group
                for t in timestamps:
                    for oid in objects:
                        if (t, oid) in predicted:
                            predicted[(t, oid)] = pid

        elif pattern_type == 'evolving convoy':
            for pid, pattern in enumerate(patterns):
                stages = pattern.stages
                for stage in stages:
                    members = stage.persistent_members.union(stage.dynamic_members)
                    timestamps = [t for t in range(stage.start_time, stage.end_time + 1)]
                    for t in timestamps:
                        members = stage.time_to_objects[t]
                        for oid in members:
                            predicted[(t, oid)] = pid

        elif pattern_type == 'convoy':
            for pid, pattern in enumerate(patterns):
                timestamps = pattern.time_to_objects.keys()

                objects = pattern.objects
                print(pid, timestamps, objects)
                for t in timestamps:
                    for oid in objects:
                        predicted[(t, oid)] = pid
        else:
            for pid, pattern in enumerate(patterns):
                t_objs = pattern.time_to_objects
                for t, oids in t_objs.items():
                    for oid in oids:
                        predicted[(t, oid)] = pid
    return {'predicted': predicted, 'truth': truth}


def save_res_to_shp(shp_path=None, saved_folder=None, res=None, dataset=None, method=None):
    predicted_res_dict = res['predicted']
    if shp_path:
        gdf = read_shapefile(shp_path)
        predicted_res_list = []
        for idx, row in gdf.iterrows():
            predicted_res_list.append(predicted_res_dict[(row.t, row.oid)])
        gdf['predicted'] = predicted_res_list
        save_shapefile(filename=rf'{saved_folder}\shps\{dataset}_{method}2.shp', data=gdf)


def save_evaluation_result_to_csv(method='proposed', F1=None, ARI=None, Precision=None, Recall=None,
                                  dataset='D1', pattern='moving cluster', t: float = -1, saved_folder=None,
                                  saved_filename='simulated_Ex_evaluation_results'):
    res = {'dataset': dataset, 'movement pattern': pattern, 'method': method, 'ARI': ARI, 'Precision': Precision,
           'Recall': Recall, 'F1': F1, 't': t}
    df = pd.json_normalize(res)
    if os.path.exists(rf'{saved_folder}\{saved_filename}.csv'):
        df.to_csv(rf'{saved_folder}\{saved_filename}.csv', header=False, index=False,
                  mode='a+')
    else:
        df.to_csv(rf'{saved_folder}\{saved_filename}.csv', index=False)


def ConvergingTree_format_predicted_truth(Patterns: List[ConvergingPattern], objects: Dict[str, MovingObject]) -> Dict[
    str, Dict]:
    predicted: Dict[Tuple[int, str], int] = {}
    truth: Dict[Tuple[int, str], int] = {}
    for oid, obj in objects.items():
        points = obj.trajectory.points
        for t, pt in points.items():
            truth[(t, oid)] = pt.truth
            predicted[(t, oid)] = -1
    if len(Patterns) == 0:
        return {'predicted': predicted, 'truth': truth}

    for pid, convergence in enumerate(Patterns):
        all_nodes = dfs_traverse_recursive(convergence.tree.root)
        print('all-node', all_nodes)
        for node in all_nodes:
            cluster = node.cluster
            for oid in cluster.members:
                predicted[(cluster.time, oid)] = pid

    return {'predicted': predicted, 'truth': truth}


def REMO_format_predicted_truth(Patterns: List[REMOConvergence], objects: Dict[str, MovingObject]) -> Dict[str, Dict]:
    predicted: Dict[Tuple[int, str], int] = {}
    truth: Dict[Tuple[int, str], int] = {}
    for oid, obj in objects.items():
        points = obj.trajectory.points
        for t, pt in points.items():
            truth[(t, oid)] = pt.truth
            predicted[(t, oid)] = -1
    if len(Patterns) == 0:
        return {'predicted': predicted, 'truth': truth}
    snapshots = SnapshotBuilder.generate_snapshots(objects)
    sorted_times = list(snapshots.keys())

    for convergence in Patterns:
        # print(convergence)
        t_start_index = sorted_times.index(convergence.t_start)

        t_end_index = sorted_times.index(convergence.t_end)
        members = convergence.members
        for i in range(t_start_index, t_end_index + 1):
            t = sorted_times[i]
            for oid in members:
                predicted[(t, oid)] = convergence.pattern_id

    return {'predicted': predicted, 'truth': truth}


def dfs_traverse_recursive(node: ConvergingTreeNode, result: List[ConvergingTreeNode] = None):
    """
    深度优先遍历（递归）
    Args:
        node: 当前节点
        result: 存储结果的列表
    Returns:
        所有节点的列表
    """
    if result is None:
        result = []
    # 访问当前节点
    result.append(node)
    print(f"访问节点 {node.level}, 簇: {node.cluster}")
    # 递归遍历所有子节点
    for child in node.children:
        dfs_traverse_recursive(child, result)
    return result


def get_cluster_labels_pred(clusters):
    traj_ids_labels = {}
    for label, cluster in clusters.items():
        for traj_id in cluster:
            traj_ids_labels[traj_id] = label

    traj_ids_labels = sorted(traj_ids_labels.items(), key=lambda x: x[0])
    labels_pred = [item[1] for item in traj_ids_labels]
    return labels_pred


def purity_index(clusters: dict, labels):
    # print(labels)
    labels_pred = get_cluster_labels_pred(clusters)
    labels = np.reshape(labels, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels[idx, :].reshape(-1)

        labels_tmp = [999 if x == -1 else x for x in labels_tmp]
        count.append(np.bincount(labels_tmp).max())
    purity = np.sum(count) / labels.shape[0]
    purity = round(purity, 4)
    # print('purity: {}'.format(purity))
    return purity


def rand_index(clusters: dict, labels):
    labels_pred = get_cluster_labels_pred(clusters)
    ARI = rand_index(labels, labels_pred)
    print(r'ARI: {}'.format(ARI))
    return ARI


# def NMI_index(clusters: dict, labels: list):
#     """
#     互信息指数, 外部指标
#     :param clusters:
#     :param labels:
#     :return:
#     """
#     labels_pred = get_cluster_labels_pred(clusters)
#
#     NMI = normalized_mutual_info_score(labels, labels_pred)
#     AMI = adjusted_mutual_info_score(labels, labels_pred)
#     NMI = round(NMI, 4)
#     AMI = round(AMI, 4)
#     # print(r'NMI: {}, AMI: {}'.format(NMI, AMI))
#     return NMI, AMI


def Adjusted_Rand_index(labels_pred, labels):
    """
    调整兰德指数, 越接近于1越好, 外部指标
    :param clusters:
    :param labels:
    :return:
    """
    ARI = adjusted_rand_score(labels, labels_pred)
    return round(ARI, 4)


def NMI_index(labels_pred, labels: list):
    """
    互信息指数, 外部指标
    :param labels_pred:
    :param labels:
    :return:
    """
    NMI = normalized_mutual_info_score(labels, labels_pred)
    AMI = adjusted_mutual_info_score(labels, labels_pred)
    NMI = round(NMI, 4)
    AMI = round(AMI, 4)
    return NMI, AMI


def entropy_index(clusters: dict, labels):
    """
    信息熵, 内部指标
    :param clusters:
    :param labels:
    :return:
    """
    n = sum([len(c) for c in clusters.values()])
    entropy = -sum([len(c) / n * np.log2(len(c) / n) for c in clusters.values()])
    entropy = round(entropy, 4)
    # print('entropy: {}'.format(entropy))
    return entropy


def recall_precision(y_true, y_pred):
    ARI = adjusted_rand_score(y_true, y_pred)
    (tn, fp), (fn, tp) = pair_confusion_matrix(y_pred, y_true)
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    # print('tn', tn, 'fp', fp, 'fn', fn, 'tp', tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, ARI
    # return {'precision': precision, 'recall': recall, 'f1': f1, 'ARI': ARI}


def get_clusters_min_distance(cluster1, cluster2, dist_matrix):
    distances = []
    for idx1 in cluster1:
        for idx2 in cluster2:
            distances.append(dist_matrix[idx1][idx2])
    return np.min(distances)


def get_cluster_max_distance(cluster, dist_matrix):
    distances = []
    for idx1, i in enumerate(cluster):
        for idx2, j in enumerate(cluster[idx1 + 1:]):
            distances.append(dist_matrix[i][j])
    return np.max(distances)


def Dunn_index(clusters: dict, dist_matrix):
    """
    :param clusters:
    :return:
    """
    centers_id = list(clusters.keys())
    cluster_max_distances = []
    for center in centers_id:
        cluster = clusters[center]
        if len(cluster) < 2:
            continue
        cluster_max_distances.append(get_cluster_max_distance(cluster, dist_matrix))
    min_inter_cluster_distances = []
    for idx1, center_id in enumerate(centers_id):
        for idx2, center_id_ in enumerate(centers_id[idx1 + 1:]):
            min_dist_i_j = get_clusters_min_distance(clusters[center_id], clusters[center_id_], dist_matrix)
            min_inter_cluster_distances.append(min_dist_i_j)
            # if min_dist_i_j < min_inter_cluster_distances[idx1]:
            #     min_inter_cluster_distances[idx1] = min_dist_i_j
            # if min_dist_i_j < min_inter_cluster_distances[idx2]:
            #     min_inter_cluster_distances[idx2] = min_dist_i_j

    dunn_index = min(min_inter_cluster_distances) / max(cluster_max_distances)
    dunn_index = round(dunn_index, 4)
    return dunn_index


def cluster_result_evaluation(clusters, labels_real, dist):
    labels_pred = get_cluster_labels_pred(clusters)
    # ARI = Adjusted_Rand_index(labels_pred, labels_real)
    Precision, Recall, F1, ARI = recall_precision(labels_pred, labels_real)
    # {'precision': precision, 'recall': recall, 'f1': f1, 'ARI': ARI}
    NMI, AMI = NMI_index(labels_pred, labels_real)
    try:
        entropy = entropy_index(clusters, labels_real)
    except:
        entropy = -999
    try:
        Dunn = Dunn_index(clusters, dist)
    except:
        Dunn = -999
    try:
        purity = purity_index(clusters, labels_real)
    except:
        purity = -999
    try:
        DBI = round(davies_bouldin_score(dist, labels_pred), 4)
    except:
        DBI = -999
    try:
        SI = silhouette_score(dist, labels_pred, metric='precomputed')
    except:
        SI = -999

    try:
        homogeneity, completeness, v_measure_score = homogeneity_completeness_v_measure(labels_real, labels_pred)
    except:
        homogeneity, completeness, v_measure_score = -999, -999, -999
    try:
        ch_score = calinski_harabasz_score(dist, labels_pred)
    except:
        ch_score = -999
    return [SI, ARI, Precision, Recall, F1, NMI, AMI, entropy, Dunn, purity, ch_score, DBI, homogeneity,
            completeness, v_measure_score]
