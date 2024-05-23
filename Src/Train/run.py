#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/15 0:30
# @Author  : name
# @File    : run.py


import Src.PathConfig as config
import Src.Model.ResNet34 as Res34
import Src.Model.ModelPara as ModelPara

import tensorflow as tf
from matplotlib import pyplot as plt
import pathlib
import pandas as pd

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

'''测试集'''
test_BatchDataset = tf.keras.utils.image_dataset_from_directory(
        config.TestDataPath,
        labels='inferred',
        label_mode='categorical',
        batch_size=8,
        image_size=(128, 128)
    )

class_name = ['beach','circularfarmland','cloud',
              'desert','forest','mountain',
              'rectangularfarmland','residential','river','snowberg']
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

    history = MODEL.fit(
            TrainData[0],
            validation_data = TrainData[1],
            steps_per_epoch=None,
            epochs=EPOCHES,
            verbose=1,
            validation_freq=1,
            workers=4,
        )

    SavePath = str(config.ResultSavePath) + f"\\{ModelName}" + "\\" + "Model.ckpt"
    MODEL.save(SavePath)

    result = MODEL.predict(test_BatchDataset, workers=4, use_multiprocessing=True)

    # 提取真实标签
    true_labels = tf.concat([y for x, y in test_BatchDataset], axis=0)
    print(true_labels)
    print(type(true_labels))
    # 将预测结果转换为类别标签
    predicted_labels = tf.argmax(result, axis=1)
    print(predicted_labels)
    print(type(predicted_labels))

    # print(result)
    # print(type(result))
    # print('ndarray的维度: ', result.ndim)
    # print('ndarray的形状: ', result.shape)
    # print('ndarray的元素数量: ', result.size)
    # print('ndarray中的数据类型: ', result.dtype)
    # for i in range(0,18000,1):
    #     index = result[i].argmax()
    #     print(class_name[index])


    # if(SaveModel == False):
    #     history = MODEL.fit(
    #         TrainData[0],
    #         validation_data = TrainData[1],
    #         steps_per_epoch=None,
    #         epochs=EPOCHES,
    #         verbose=1,
    #         validation_freq=1,
    #         workers=4,
    #
    #     )
    # else:
    #     if(SaveModelFormat == "h5"):
    #         CheckpointSavePath = str(config.ResultSavePath) + f"\\{ModelName}" + "\\" + "{epoch:02d}-{val_categorical_accuracy:.2f}.h5"
    #         if(SaveWeightsOnly == True):
    #             cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CheckpointSavePath,
    #                                                              monitor='val_categorical_accuracy',
    #                                                              mode='max',
    #                                                              save_weights_only=True
    #                                                              )
    #         else:
    #                 print("NotImplementedError: Saving the model to HDF5 format requires the model to be a Functional model or a Sequential model.")
    #                 return
    #     elif(SaveModelFormat == "ckpt"):
    #         CheckpointSavePath = str(config.ResultSavePath) + f"\\{ModelName}" + "\\" + "{epoch:02d}-{val_categorical_accuracy:.2f}.ckpt"
    #         if (SaveWeightsOnly == True):
    #             cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CheckpointSavePath,
    #                                                              monitor='val_categorical_accuracy',
    #                                                              mode='max',
    #                                                              save_weights_only=True
    #                                                              )
    #         else:
    #             cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CheckpointSavePath,
    #                                                              monitor='val_categorical_accuracy',
    #                                                              mode='max',
    #                                                              save_weights_only=False
    #                                                              )
    #     else:
    #         print("Model save format error")
    #         return
    #
    #     # CheckpointSavePath = str(config.ResultSavePath) +"\\" +ModelName + "\\" + "{epoch:02d}-{val_categorical_accuracy:.2f}.ckpt"
    #     # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CheckpointSavePath,
    #     #                                                  monitor='val_categorical_accuracy',
    #     #                                                  mode='max',
    #     #                                                  save_weights_only=True
    #     #                                                  )
    #     history = MODEL.fit(
    #         TrainData[0],
    #         validation_data=TrainData[1],
    #         steps_per_epoch=None,
    #         epochs=EPOCHES,
    #         verbose=1,
    #         # validation_split=0.1,
    #         validation_freq=1,
    #         callbacks=[cp_callback],
    #         workers=4
    #     )
    # # ValueError: `validation_split` is only supported for Tensors or NumPy arrays, found following types in the input: [<class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>]
    #
    # MODEL.summary()



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
        accuracy = history.history["categorical_accuracy"]
        val_loss = history.history["val_loss"]
        val_accuracy = history.history["val_categorical_accuracy"]
        x = range(1, EPOCHES + 1, 1)

        fig, ax = plt.subplots(figsize=(10,5))
        # fig = plt.figure(figsize=(10,5))
        # ax = fig.add_subplot(111)
        ax.plot(x,loss,'-',color='#FF0000',label = "Loss")
        ax.plot(x, val_loss, '-', color='#FFA500', label="Val_Loss")
        ax2=ax.twinx()
        ax2.plot(x,accuracy,'-',color='blue',label = "Accuracy")
        ax2.plot(x, val_accuracy, '-',color='green', label="Val_Accuracy")

        ax.set_xlabel("epoch")
        ax.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")

        fig.legend(loc=(0.5,0),ncol=4)

        plt.savefig(fname=pltsave_str,bbox_inches='tight')
        plt.show()

