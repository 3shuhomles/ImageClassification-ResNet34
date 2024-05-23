#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/24 0:39
# @Author  : name
# @File    : test03.py

import Src.PathConfig as config
import Src.Model.ResNet34 as Res34
import Src.Model.ModelPara as ModelPara

import tensorflow as tf
from matplotlib import pyplot as plt
import pathlib

Model = Res34

'''测试集'''
train_BatchDataset = tf.keras.utils.image_dataset_from_directory(
    config.TestDataPath,
    labels='inferred',
    label_mode='categorical',
    batch_size=8,
    image_size=(128,128)
)

model = tf.keras.models.load_model(Model,filepath = r'E:\code\githubWorkSpace\ImageClassification-ResNet34\result\UndefinedModel\Model.ckpt')




