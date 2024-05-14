import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import os

# GPU配置
os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#读入数据
train_x = np.array([[1.4, 0.2],
                        [1.7, 0.4],
                        [1.5, 0.4],
                        [2.3, 0.7],
                        [2.7, 1.1],
                        [2.6, 0.9],
                        [4.6, 1.3],
                        [3.5, 1.0],
                        [3.9, 1.2]])
train_y = np.array([[1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1]])
#搭建模型
model = Sequential()
model.add(Dense(units = 2, input_dim = 2))
#model.add(Dense(units = 2, input_dim = 2, activation = 'sigmoid'))
model.add(Dense(units = 3, activation = 'softmax'))
#编译模型
model.compile(optimizer = 'adam', loss = 'mse')
#训练模型
history = model.fit(x = train_x, y = train_y, epochs = 1000)
