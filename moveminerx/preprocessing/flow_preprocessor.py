#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: flow_preprocessor.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 14:53

from moveminerx.preprocessing.base import BasePreprocessor
from shapely.geometry import LineString


class FlowPreprocessor(BasePreprocessor):
    """
    Flow 预处理，包括：
    - Flow cleaning
    - OD extraction
    - Map matching
    """

    def clean(self, flows):
        """
        移除异常flow（长度为0或volume为0）
        """
        return [f for f in flows if f.geometry.length > 0 and f.volume > 0]

    def extract_od(self, flow):
        """
        OD提取
        """
        origin = flow.geometry.coords[0]
        destination = flow.geometry.coords[-1]
        return origin, destination

    def map_match(self, flow, map_data=None):
        """
        map matching placeholder
        """
        # TODO: 调用HMM或其他map matching库
        return flow

    def preprocess(self, flows):
        processed = self.clean(flows)
        processed = [self.map_match(f) for f in processed]
        return processed