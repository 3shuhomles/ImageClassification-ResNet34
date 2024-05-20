#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/15 0:30
# @Author  : name
# @File    : run.py


import Src.PathConfig as config
import Src.Model.ResNet34 as Res34
import Src.Model.ModelPara as ModelPara

import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pathlib

'''训练集'''
train_BatchDataset = tf.keras.utils.image_dataset_from_directory(
    config.TrainDataPath,
    labels='inferred',
    label_mode='categorical',
    batch_size=8,
    image_size=(128,128),
    shuffle=True,
    seed = ModelPara.SEED,  # 随机种子 between 0 and 2**32-1
    # shuffle=False,
    subset='both',
    validation_split=0.1,    # 十折交叉验证

    # shuffle=False,
    # subset='training',
    # validation_split=0.1
)
# print(train_BatchDataset)
# print(type(train_BatchDataset))
# print(type(train_BatchDataset[0]))
# print(type(train_BatchDataset[1]))
# ValueError: If using `validation_split` and shuffling the data, you must provide a `seed` argument,
#       to make sure that there is no overlap between the training and validation subset.
# ValueError: If `subset` is set, `validation_split` must be set, and inversely

# matplotlib支持中文设置
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False

def ModelFit(EPOCHES,ModelName = "UndefinedModel",MODEL = Res34,TrainData = train_BatchDataset,SaveFigure = False,SaveModel = False,SaveModelFormat = "h5",SaveWeightsOnly = True):
    if(pathlib.Path(str(config.ResultSavePath) + f"\\{ModelName}").exists()):
        print("ModelName is exists")
        return
    else:
        pathlib.Path(str(config.ResultSavePath) + f"\\{ModelName}").mkdir()


    if(SaveModel == False):
        history = MODEL.fit(
            TrainData[0],
            validation_data = TrainData[1],
            steps_per_epoch=None,
            epochs=EPOCHES,
            verbose=1,
            validation_freq=1,
            workers=4,

        )
    else:
        if(SaveModelFormat == "h5"):
            CheckpointSavePath = str(config.ResultSavePath) + f"\\{ModelName}" + "\\" + "{epoch:02d}-{val_categorical_accuracy:.2f}.h5"
            if(SaveWeightsOnly == True):
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CheckpointSavePath,
                                                                 monitor='val_categorical_accuracy',
                                                                 mode='max',
                                                                 save_weights_only=True
                                                                 )
            else:
                    print("NotImplementedError: Saving the model to HDF5 format requires the model to be a Functional model or a Sequential model.")
                    return
        elif(SaveModelFormat == "ckpt"):
            CheckpointSavePath = str(config.ResultSavePath) + f"\\{ModelName}" + "\\" + "{epoch:02d}-{val_categorical_accuracy:.2f}.ckpt"
            if (SaveWeightsOnly == True):
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CheckpointSavePath,
                                                                 monitor='val_categorical_accuracy',
                                                                 mode='max',
                                                                 save_weights_only=True
                                                                 )
            else:
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CheckpointSavePath,
                                                                 monitor='val_categorical_accuracy',
                                                                 mode='max',
                                                                 save_weights_only=False
                                                                 )
        else:
            print("Model save format error")
            return

        # CheckpointSavePath = str(config.ResultSavePath) +"\\" +ModelName + "\\" + "{epoch:02d}-{val_categorical_accuracy:.2f}.ckpt"
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CheckpointSavePath,
        #                                                  monitor='val_categorical_accuracy',
        #                                                  mode='max',
        #                                                  save_weights_only=True
        #                                                  )
        history = MODEL.fit(
            TrainData[0],
            validation_data=TrainData[1],
            steps_per_epoch=None,
            epochs=EPOCHES,
            verbose=1,
            # validation_split=0.1,
            validation_freq=1,
            callbacks=[cp_callback],
            workers=4
        )
    # ValueError: `validation_split` is only supported for Tensors or NumPy arrays, found following types in the input: [<class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>]

    MODEL.summary()

    if(SaveFigure == True):
        pltsave_str = str(config.ResultSavePath) + "\\" +ModelName + "\\Loss_and_Accuracy.png"
        pltsave_str_2 = str(config.ResultSavePath) + "\\" +ModelName + "\\Val_Loss_and_Accuracy.png"
        '''history曲线'''
        # pd.DataFrame(history.history).plot(figsize=(10, 5))
        # plt.grid(True)
        # plt.gca().set_ylim(0, 1.2)
        # plt.xlabel("迭代次数")
        # plt.xticks(np.arange(1, EPOCHES + 1, 1))
        # plt.savefig(fname=pltsave_str)
        # print('history图像保存成功')
        # plt.show()

        loss = history.history["loss"]
        print(history.history)
        print(type(history.history))
        print(history.history["loss"])
        print(type(history.history["loss"]))
        print(history.history.keys)
        accuracy = history.history["categorical_accuracy"]
        val_loss = history.history["val_loss"]
        val_accuracy = history.history["val_categorical_accuracy"]
        x = range(1, EPOCHES + 1, 1)

        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.plot(x,loss,'-',color='#FF0000',label = "Loss")
        ax2=ax.twinx()
        ax.plot(x, val_loss, '-',color='#FFA500', label="Val_Loss")
        ax2.plot(x,accuracy,'-',color='blue',label = "Accuracy")
        ax2.plot(x, val_accuracy, '-',color='green', label="Val_Accuracy")
        plt.legend()

        ax.set_xlabel("epoch")
        ax.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")

        plt.savefig(fname=pltsave_str)
        plt.show()

