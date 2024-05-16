#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/14 23:36
# @Author  : name
# @File    : ResNet34.py

from Src.Model.ResNet import ResNet
import tensorflow as tf

from Src.Model.ModelPara import LEARNINGRATE_1

ResNet34 = ResNet([3, 4, 6, 3])
ResNet34.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(), # BinaryCrossentropy
    optimizer=tf.keras.optimizers.Adam(learning_rate = LEARNINGRATE_1),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)
