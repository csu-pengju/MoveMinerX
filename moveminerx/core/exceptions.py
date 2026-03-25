#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Ju
# @File Name: exceptions.py
# @E-mail: daisy_pj@csu.edu.cn
# @Time of Creation: 2026/3/25 12:16




class MoveMinerXError(Exception):
    """Base exception"""
    pass


class NotFittedError(MoveMinerXError):
    """Model not fitted"""
    pass


class RegistryError(MoveMinerXError):
    """Registry related error"""
    pass
