#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/14 23:36
# @Author  : name
# @File    : ResNet34.py

from Src.Model.ResNet import ResNet
import Src.Model.ModelPara as ModelPara

import tensorflow as tf


ResNet34 = ResNet([3, 4, 6, 3])
ResNet34.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(), # BinaryCrossentropy
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
    # optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)

ResNet18 = ResNet([2, 2, 2, 2])
ResNet18.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(), # BinaryCrossentropy
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)

