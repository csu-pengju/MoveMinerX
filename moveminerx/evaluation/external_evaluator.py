#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: external_evaluator.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 15:54
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from moveminerx.evaluation.base import BaseEvaluator


class ExternalEvaluator(BaseEvaluator):
    """
    外部指标评估
    """
    def __init__(self, average='macro'):
        self.average = average

    def evaluate(self, patterns, ground_truth):
        """
        patterns: list of cluster labels or pattern assignments
        ground_truth: same format as patterns
        """
        # 将模式转换为标签
        # 假设 patterns 和 ground_truth 都是长度 N 的列表
        y_pred = patterns
        y_true = ground_truth

        metrics = {
            'precision': precision_score(y_true, y_pred, average=self.average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=self.average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=self.average, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'ARI': adjusted_rand_score(y_true, y_pred),
            'AMI': adjusted_mutual_info_score(y_true, y_pred)
        }
        return metrics