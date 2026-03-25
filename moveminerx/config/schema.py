#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: schema.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 12:21
# moveminerx/config/schema.py

REQUIRED_FIELDS = {
    "mining": ["method", "params"],
    "data": ["input_path"]
}


def validate_config(cfg: dict):
    """
    简单配置校验
    """
    for section, fields in REQUIRED_FIELDS.items():
        if section not in cfg:
            raise ValueError(f"Missing section: {section}")

        for field in fields:
            if field not in cfg[section]:
                raise ValueError(
                    f"Missing field '{field}' in section '{section}'"
                )