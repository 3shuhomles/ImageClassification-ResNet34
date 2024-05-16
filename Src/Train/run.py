#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/15 0:30
# @Author  : name
# @File    : run.py

import tensorflow as tf
import Src.Train.config as config
import Src.Model.ResNet34 as Res34
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pathlib

'''训练集'''
train_BatchDataset = tf.keras.utils.image_dataset_from_directory(
    config.TrainDataPath,
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

# matplotlib支持中文设置
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False

def ModelFit(EPOCHES,ModelName = "UndefinedModel",MODEL = Res34,TrainData = train_BatchDataset,SaveCheckpoint = False,SaveFigure = True):
    CheckpointSavePath = str(config.CheckpointSaveDir) + "\\" + ModelName + ".ckpt"
    pltsave_str = str(config.FigureSavePath) + f"\\Loss_and_CategoricalAccuracy_{ModelName}.png"
    if(pathlib.Path(CheckpointSavePath).exists()):
        print("ModelName is exists")
        return
    if(pathlib.Path(pltsave_str).exists()):
        print("ModelName is exists")
        return

    if(SaveCheckpoint == False):
        history = MODEL.fit(
            TrainData,
            steps_per_epoch=None,
            epochs=EPOCHES,
            verbose=1,
            validation_freq=1,
            workers=4
        )
    else:
        CheckpointSavePath = str(config.CheckpointSaveDir) + "\\" +ModelName + ".ckpt"
        history = MODEL.fit(
            TrainData,
            steps_per_epoch=None,
            epochs=EPOCHES,
            verbose=1,
            validation_freq=1,
            callbacks=[CheckpointSavePath],
            workers=4
        )

    MODEL.summary()

    if(SaveFigure == True):
        pltsave_str = str(config.FigureSavePath) + f"\\Loss_and_CategoricalAccuracy_{ModelName}.png"
        '''history曲线'''
        pd.DataFrame(history.history).plot(figsize=(10, 5))

        plt.grid(True)
        plt.gca().set_ylim(0, 1.2)
        plt.xlabel("迭代次数")
        plt.xticks(np.arange(1, EPOCHES + 1, 1))
        plt.savefig(fname=pltsave_str)
        print('history图像保存成功')
        plt.show()




