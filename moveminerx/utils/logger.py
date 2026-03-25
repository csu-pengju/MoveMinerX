#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: logger.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 12:28
# moveminerx/utils/logger.py

import logging
import sys


def get_logger(name="MoveMinerX", level=logging.INFO):
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # 防止重复添加handler

    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def add_file_handler(logger, file_path):
    file_handler = logging.FileHandler(file_path)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)