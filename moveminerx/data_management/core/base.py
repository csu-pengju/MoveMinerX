#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: base.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 13:36

class BaseData:
    """
    所有数据类型的基类
    """
    def __init__(self, obj_id):
        self.obj_id = obj_id
