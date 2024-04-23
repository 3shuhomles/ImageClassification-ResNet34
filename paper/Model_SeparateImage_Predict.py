
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from keras import Model

# # GPU配置
os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 0为独显，1为集显 会报错 自动跳转CPU

'''config'''
EPOCHES = 30
LEARNINGRATE = 0.0001
# generation = '2nd'
NAME = f'ResNet34_{EPOCHES}_{LEARNINGRATE}'
# NAME = f'ResNet34_{EPOCHES}_{LEARNINGRATE}_{generation}'
# NAME = f'ResNet34_{EPOCHES}_{LEARNINGRATE}_Notshuffle'

# 加载模型
load_path = f"./HDF5_{NAME}"
# load_path = f"E:\ResNet34\HDF5_{NAME}"
model_2 = tf.keras.models.load_model(load_path)

# 测试图像路径
img_path = (r'D:\code\paper\paper\dataset\TG2\test\beach\beach_1901.jpg')
img = plt.imread(img_path)
# print(type(img))

# 图像TF格式变换-->NHWC
img = np.expand_dims(img, axis=-1)
img = np.rollaxis(img,3,0)
img = tf.constant(img, dtype=tf.float32)
# print(img.shape)    # NHWC

class_name = ['beach','circularfarmland','cloud',
              'desert','forest','mountain',
              'rectangularfarmland','residential','river','snowberg']
predict_activefunction = model_2.predict(img)
# print(predict_activefunction,type(predict_activefunction),predict_activefunction.shape)

print('softmax函数结果：')
print(predict_activefunction)
print('-'*30)

# 最大softmax结果按最大值索引降序排列数组
lst = np.argsort(predict_activefunction[0])[::-1]
print('softmax函数结果最大值索引排序数组：')
print(lst)
print('-'*30)

print('该图像预测分类为：',class_name[lst[0]])

