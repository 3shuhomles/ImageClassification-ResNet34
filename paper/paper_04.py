# 导入包
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from keras import Model

import matplotlib.pyplot as plt

# GPU配置
os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 设置显示格式
np.set_printoptions(threshold=np.inf)

# '''导入数据集'''
# fashion = tf.keras.datasets.fashion_mnist
# (x_train, y_train), (x_validation, y_validation) = fashion.load_data()

# print(x_train.shape)
# print(type(x_train))
# exmple = x_train[0,:,:]
# # print(exmple)
# print(exmple.shape)
# plt.imshow(exmple)
# plt.show()

# x_train, x_validation = x_train / 255.0, x_validation / 255.0
# print("x_train.shape", x_train.shape)
# # 给数据增加一个维度，使数据和网络结构匹配
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_validation = x_validation.reshape(x_validation.shape[0], 28, 28, 1)
# print("x_train.shape", x_train.shape)
#
# # 缩小数据集
# x_train = x_train[:60]
# y_train = y_train[:60]
# x_test = x_validation[20:25]
# y_test = y_validation[20:25]
# x_validation = x_validation[:10]
# y_validation = y_validation[:10]
# print("x_train.shape", x_train.shape)
# print("y_train.shape", y_train.shape)
# print("x_test.shape", x_test.shape)
# print("y_test.shape", y_test.shape)
# print("x_validation.shape", x_validation.shape)
# print("y_validation.shape", y_validation.shape)
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(10, activation="softmax"),
# ])
#
# model.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(), # BinaryCrossentropy
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
# )
#
# history = model.fit(
#     x_train, y_train,
#     steps_per_epoch=30,
#     epochs=10,
#     validation_data=(x_validation, y_validation),
#     validation_freq=1,
#     verbose=2,
#     # callbacks=[tf_callback]
# )
#
# # 显示结果
# model.summary()
#
# 保存模型
# save_path = "./HDF5_test"
# model.save(save_path)
# print('Model保存成功')


# # 在测试集上测试
# print(model.evaluate(x_test, y_test))
# X_new=x_test[:3]
# Y_proba= model.predict(X_new)
# print(Y_proba.round(2))
# y_pre=model.predict(X_new)
# print(y_pre)

model_path = "./HDF5_test"
model = tf.keras.models.load_model(model_path)

'''全部层的参数列表'''
v = model.trainable_variables
print(type(v))
print(len(v))
for i in range(0,len(v)):
    # print(type(i))
    print(v[i].name)
    print(v[i].shape)
    print('-'*20)
    # print(type(i.numpy()))
    if i == 0:
    # 获取第一个卷积层的参数
        con = v[i].numpy()
print(type(con))
print(con.shape)
print(len(con.shape))
# # 获取第一个卷积层的第一个卷积
# exmple = con[:,:,:,0]
# # print(exmple)
# print(exmple.shape)
# exmple = np.rollaxis(exmple,2,0)
# print(exmple)
# print(exmple.shape)
# print('-'*30)
'''RGB图像ndarray格式的输出可读化'''
# exmple_2 = con
# exmple_2 = np.rollaxis(exmple_2,2,0)
# exmple_2 = np.rollaxis(exmple_2,3,0)
# print(exmple_2)
# print(exmple_2.shape)
# print('over!!!')


'''读取文件夹下的照片'''
# dirpath = r'.\dataset\classification\train\otter\otter_10001.jpg'
# img_plt = plt.imread(dirpath)
# print("img_plt :",img_plt .shape)
# print("img_plt :",type(img_plt ))
# 可选择通道画出
# plt.imshow(img_plt[:,:,:])
# , cmap=plt.cm.binary
# plt.show()

'''数组可直接绘图'''
# x = np.random.randint(0,255,[100,100,3])
# plt.imshow(x)
# plt.show()

'''换轴'''
# x = np.random.randint(0,3,[3,2,4,5])
# print(x)
# print(x.shape)
# print(x.ndim)
# # print(x.strides)
# # print(x.nbytes)
# # print(x.itemsize)
# print('-'*30)
# x_1 = np.rollaxis(x,3,0)
# print(x_1.shape)
# y = np.random.randint(0,3,[3,4])
# print(y)
# y_1 = np.rollaxis(y,1,0)
# print(y_1)

'''保存权重参数'''
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     if len(v.shape) == 4:
#         v_numpy = np.rollaxis(v.numpy(),2,0)
#         v_numpy = np.rollaxis(v_numpy,3,0)
#     file.write(str(v_numpy) + '\n')
# file.close()






