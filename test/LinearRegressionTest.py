#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/3 2:12
# @Author  : name
# @File    : LinearRegressionTest.py

import numpy as np
import tensorflow as tf

def LRTest():
    try:
        X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
        y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

        X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
        y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
        X = tf.constant(X)
        y = tf.constant(y)

        a = tf.Variable(initial_value=0.)
        b = tf.Variable(initial_value=0.)
        variables = [a, b]

        '''
        Model：Y = aX + b，X、Y是一维张量（向量）
        Cost Function：(Pre_Y - Y)^2 平方和误差函数（区别于平均平方和误差函数）
        '''

        num_epoch = 3000
        optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4) # 优化器，梯度下降法
        for e in range(num_epoch):
            # if((e / 1000) in range(1,11,1)):
            #     print(f"epoch:{e}")

            # 使用tf.GradientTape()记录损失函数的梯度信息
            with tf.GradientTape() as tape:
                y_pred = a * X + b # 模型
                loss = tf.reduce_sum(tf.square(y_pred - y)) # 损失函数、成本函数
            # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
            grads = tape.gradient(loss, variables)
            # TensorFlow自动根据梯度更新参数
            optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
        # print(a)
        # print(b)
        print("LRTestResult: PASS")
    except:
        print("LRTestResult: FAILED")





