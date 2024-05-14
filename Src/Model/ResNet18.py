#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/14 23:51
# @Author  : name
# @File    : ResNet18.py

from ResNet import ResNet
import tensorflow as tf

from ModelPara import LEARNINGRATE_1

ResNet18 = ResNet([2, 2, 2, 2])
ResNet18.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(), # BinaryCrossentropy
    optimizer=tf.keras.optimizers.Adam(learning_rate = LEARNINGRATE_1),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)
