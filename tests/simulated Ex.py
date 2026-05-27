#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: simulated Ex.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/4/21 21:31
import time

from moveminerx.mining._basic_class import TrajectoryLoader, MovingObject
from moveminerx.mining.accompanying._convoy import CMCAlgorithm
from moveminerx.mining.accompanying._evolving_convoy import EvolvingConvoysMiner, SimpleSliceBySlice
from moveminerx.mining.accompanying._loose_travelling_companion import LooseTravelingCompanionMiner
from moveminerx.mining.convergence._REMO_based_ConvergenceDetection import ConvergenceDetector
from moveminerx.mining.convergence._convergingDetection import ConvergingPatternMiner
from tests.evaluations import format_predicted_truth, match_pattern_truth, save_res_to_shp, \
    save_evaluation_result_to_csv, ConvergingTree_format_predicted_truth, REMO_format_predicted_truth
from util import read_shapefile, save_shapefile
from moveminerx.mining.accompanying._moving_cluster import MCAlgorithm


def test_accompanying_pattern():
    data_folder = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\datasets\accompanying patterns'
    data_shp = read_shapefile(rf'{data_folder}\AcD6_pro.shp')
    # geos = data_shp['geometry'].values.tolist()
    dataset = 'AcD6'
    method = 'mc1'
    objects = TrajectoryLoader.load_from_shp(rf'{data_folder}\AcD6_pro.shp')
    t1 = time.time()
    # Moving cluster
    movingClusterMiner = MCAlgorithm(objects=objects, k=2, theta=0.8, eps=20, minPts=2, alpha=0.05,
                                     clustering_metric='precomputed')
    moving_clusters = movingClusterMiner.MC1()
    t2 = time.time()

    res = format_predicted_truth(patterns=moving_clusters, objects=objects)
    eval_res = match_pattern_truth(predicted=res['predicted'], truth=res['truth'])
    print(eval_res)
    # save_res_to_shp(shp_path=rf'{data_folder}\AcD6_pro.shp', res=res, saved_folder=saved_folder,
    #                 dataset=dataset, method=method + ' mc1')
    save(eval_res=eval_res, method=method, dataset=dataset, t=t2-t1, pattern='moving cluster')

    # Convoy
    method = 'Convoy_CMC'
    t1 = time.time()
    convoyMiner = CMCAlgorithm(objects=objects, k=2, m=2, eps=20, min_pts=2, metric='precomputed')
    convoys = convoyMiner.run()
    t2 = time.time()
    res = format_predicted_truth(patterns=convoys, objects=objects, pattern_type='convoy')
    eval_res = match_pattern_truth(predicted=res['predicted'], truth=res['truth'])
    print(eval_res)
    save(eval_res=eval_res, method=method, dataset=dataset, t=t2 - t1, pattern='convoy')

    # Loose traveling companion
    t1 = time.time()
    method = 'LTC_SF'
    looseTravelingCompanionMiner = LooseTravelingCompanionMiner(objects=objects, mG=3, dG=3, fC=1,
                                                                eps=20, min_pts=2, max_group_size=20)
    loose_traveling_companions = looseTravelingCompanionMiner.run()
    t2 = time.time()
    res = format_predicted_truth(patterns=loose_traveling_companions, objects=objects, pattern_type='loose traveling companion')
    eval_res = match_pattern_truth(predicted=res['predicted'], truth=res['truth'])
    save(eval_res=eval_res, method=method, dataset=dataset, t=t2 - t1, pattern='loose traveling companion')

    print(eval_res)

    # Evolving Convoys
    method = r'EConvoy_S3'
    t1 = time.time()
    evolvingConvoysMiner = SimpleSliceBySlice(objects=objects, m=3, k=3, w=4, eps=20, minPts=2)
    evolving_convoys = evolvingConvoysMiner.run()
    t2 = time.time()
    res = format_predicted_truth(patterns=evolving_convoys, objects=objects, pattern_type='evolving convoy')
    eval_res = match_pattern_truth(predicted=res['predicted'], truth=res['truth'])
    save(eval_res=eval_res, method=method, dataset=dataset, t=t2 - t1, pattern='evolving convoy')
    print(eval_res)


def save(eval_res, method, dataset, t, pattern):
    f1 = eval_res['f1']
    recall = eval_res['recall']
    precision = eval_res['precision']
    ARI = eval_res['ARI']
    save_evaluation_result_to_csv(method=method, F1=f1, Precision=precision, ARI=ARI,
                                  pattern=pattern, Recall=recall, dataset=dataset, t=t, saved_folder=saved_folder)


def test_convergence_pattern():
    data_folder = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\datasets\convergence patterns'
    data_shp = read_shapefile(rf'{data_folder}\CD4_pro.shp')
    # geos = data_shp['geometry'].values.tolist()

    dataset = 'CD4'
    objects = TrajectoryLoader.load_from_shp(rf'{data_folder}\CD4_pro.shp')
    t1 = time.time()
    # cluster growth-based method
    method = 'cluster growth-based method'
    convergenceTreeMiner = ConvergingPatternMiner(objects=objects, k_t=3, k_m=3, k_p=2, eps=10, minPts=2)
    convergences = convergenceTreeMiner.discovery_converging_patterns()
    t2 = time.time()
    res = ConvergingTree_format_predicted_truth(convergences, objects=objects)
    eval_res = match_pattern_truth(predicted=res['predicted'], truth=res['truth'])
    save(eval_res=eval_res, method=method, dataset=dataset, t=t2 - t1, pattern='convergence')

    # REMO based method
    method = 'direction vector-based method'
    remo_convergenceMiner = ConvergenceDetector(objects=objects)
    convergences = remo_convergenceMiner.detect_convergence_serial(t_start=1, t_end=60,
                                                                   interval_length=59,
                                                                   r=50, m_threshold=3, dataset=dataset)
    res = REMO_format_predicted_truth(convergences, objects=objects)
    eval_res = match_pattern_truth(predicted=res['predicted'], truth=res['truth'])
    # save_res_to_shp(shp_path=rf'..\simulated Ex\data\{dataset}.shp', res=res, dataset=dataset, method=method)
    save(eval_res=eval_res, method=method, dataset=dataset, t=t2 - t1, pattern='convergence')


def test_point_aggregation_pattern():

    pass


def test_trajectory_aggregation_pattern():
    pass


def test_flow_aggregation_pattern():
    pass


if __name__ == "__main__":
    saved_folder = r'E:\OneDrive\成果\01-论文\21-运动模式挖掘工具箱\results'
    # test_accompanying_pattern()
    test_convergence_pattern()
