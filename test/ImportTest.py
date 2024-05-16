#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/14 23:27
# @Author  : name
# @File    : ImportTest.py

import tensorflow as tf

def ImTest():
    try:
        print(tf.__version__)
        print("ImTestResult: PASS")
    except:
        print("ImTestResult: FAILED")

