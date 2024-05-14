# 导入包
# import os
import numpy as np
# import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
# from matplotlib import pyplot as plt
# from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
# from keras import Model

# 仔细分析控制台展示的三维数组结构
x = np.random.randint(0,3,[3,4,2,5])
print(x.shape)
print(x.ndim)
print(x)
print('-'*30)
# x = np.rollaxis(x,2,0)
# print(x.shape)
# print(x.ndim)
# print(x)

print('-'*50)
y = range(1,1001)
print(y[20:25])












