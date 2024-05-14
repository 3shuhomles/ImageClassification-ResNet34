#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/15 0:30
# @Author  : name
# @File    : run.py

import tensorflow as tf
import config

'''训练集'''
train_BatchDataset = tf.keras.utils.image_dataset_from_directory(
    config.TrainPDataPath,
    labels='inferred',
    label_mode='categorical',
    batch_size=5,
    image_size=(128,128),
    # shuffle=True,
    seed = 1635,  # 随机种子 between 0 and 2**32-1
    shuffle=True,
    subset='training',
    validation_split=0.1    # 十折交叉验证

    # shuffle=False,
    # subset='training',
    # validation_split=0.1
)



def ModelFit(EPOCHES,MODEL,TrainData):
    history = MODEL.fit(
        train_BatchDataset,
        steps_per_epoch=None,
        epochs=EPOCHES,
        verbose=1,
        validation_freq=1,
        callbacks=[cp_callback],
        workers=4
    )

    MODEL.summary()
    pass