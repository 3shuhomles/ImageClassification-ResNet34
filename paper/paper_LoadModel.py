
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from keras import Model

# # GPU配置
# os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 设置显示格式
np.set_printoptions(threshold=np.inf)

'''导入数据集'''
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_validation, y_validation) = fashion.load_data()
x_train, x_validation = x_train / 255.0, x_validation / 255.0
print("x_train.shape", x_train.shape)
# 给数据增加一个维度，使数据和网络结构匹配
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_validation = x_validation.reshape(x_validation.shape[0], 28, 28, 1)
print("x_train.shape", x_train.shape)
# 缩小数据集
x_train = x_train[:60]
y_train = y_train[:60]
x_test = x_validation[20:25]
y_test = y_validation[20:25]
x_validation = x_validation[:10]
y_validation = y_validation[:10]
print("x_train.shape", x_train.shape)
print("y_train.shape", y_train.shape)
print("x_test.shape", x_test.shape)
print("y_test.shape", y_test.shape)
print("x_validation.shape", x_validation.shape)
print("y_validation.shape", y_validation.shape)


load_path = "./HDF5"
model_2 = tf.keras.models.load_model(load_path)

# 与原模型预测做对比，是否一致
print(model_2.evaluate(x_test, y_test))
X_new=x_test[:3]

# Y_proba_2= model_2.predict(X_new)
# print(Y_proba_2.round(2))

y_pre_2=model_2.predict(X_new)
print(y_pre_2)
print(type(y_pre_2))





