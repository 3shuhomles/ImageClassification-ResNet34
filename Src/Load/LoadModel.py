#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/8 22:27
# @Author  : name
# @File    : test04.py

import tensorflow as tf

import Src.PathConfig as config
import Src.Model.ResNet34 as Res34
from Src.Model.ResNet import ResNet
import Src.Model.ModelPara as ModelPara

# tf格式 模型
Model_tf_path = r'E:\code\githubWorkSpace\ImageClassification-ResNet34\result\Res34_TrainData1800_AdaMax_0.001_epoch30\ModelSave'
MODEL = tf.keras.models.load_model(Model_tf_path)

# h5格式 模型权重
# Weights_h5_path = r'E:\code\githubWorkSpace\ImageClassification-ResNet34\result\save_weights_h5\WeightsSave\ModelWeights'
# MODEL = Res34
# MODEL.load_weights(Weights_h5_path)

# tf格式 模型权重
# Weights_tf_path = r'E:\code\githubWorkSpace\ImageClassification-ResNet34\result\Res34_TrainData1800_AdaMax_0.001_epoch30\WeightsSave\ModelWeights.index'
# MODEL = Res34
# MODEL.load_weights(Weights_tf_path)


'''测试集'''
test_BatchDataset = tf.keras.utils.image_dataset_from_directory(
        config.TestDataPath,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=(128, 128),
        shuffle=False
    )

# 提取真实标签
true_labels = tf.concat([y for x, y in test_BatchDataset], axis=0)
result = MODEL.predict(test_BatchDataset)
# print(result)
'''
因为模型是10分类，故返回结果则是一个1*10的向量，其中每列的值表示该图像属于该类的概率；
result维度：传入图像数（18000）*分类数目（10）
'''
accuracy = tf.keras.metrics.CategoricalAccuracy()
accuracy.update_state(true_labels, result)
print(f"准确率：{accuracy.result().numpy()}")
print(accuracy.result().numpy())