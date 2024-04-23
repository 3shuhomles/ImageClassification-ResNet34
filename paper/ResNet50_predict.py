
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
EPOCHES = 5
LEARNINGRATE = 0.001

NAME = f'ResNet50_{EPOCHES}_{LEARNINGRATE}'


# 加载模型
load_path = f"./HDF5_{NAME}"

model = tf.keras.models.load_model(load_path)

# 测试图像路径
img_path = (r'D:\code\paper\paper\dataset\TG2\train\snowberg\snowberg_0084.jpg')
img = plt.imread(img_path)
print(type(img),img.shape)    #<class 'numpy.ndarray'> (128, 128, 3)

# # 图像TF格式变换-->NHWC
img = np.expand_dims(img, axis=-1)
img = np.rollaxis(img,3,0)
img = tf.constant(img, dtype=tf.float32)
print(type(img),img.shape)    # NHWC
# <class 'tensorflow.python.framework.ops.EagerTensor'> (1, 128, 128, 3)

img = tf.image.resize(img,(224,224))
print(type(img),img.shape)
# <class 'tensorflow.python.framework.ops.EagerTensor'> (224, 224, 3)

class_name = ['beach','circularfarmland','cloud',
              'desert','forest','mountain',
              'rectangularfarmland','residential','river','snowberg']

# predict_activefunction = model.predict(img)
# print(predict_activefunction,type(predict_activefunction),predict_activefunction.shape)

# 最大softmax结果按最大值索引降序排列数组
# lst = np.argsort(predict_activefunction[0])[::-1]
# # print(lst,type(lst),lst.shape)
# print('该图像预测分类为：',class_name[lst[0]])


# for layer in model.layers:
#     print(layer.name)
#     print(type(layer))
# layer = model.get_layer('predictions')
# print(layer.output)


'''feature map'''
layer_outputs = [layer.output for layer in model.layers] #前16层输出
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
#
activations = activation_model.predict(img) #12组特征层输出
#
# plt.matshow(activations[1][0,:,:,0], cmap='viridis') #第1卷积层的第1特征层输出
# plt.matshow(activations[1][0,:,:,1], cmap='viridis') #第1卷积层的第0特征层输出
#
# plt.show()

layer_names = []
for layer in model.layers:
    layer_names.append(layer.name) #特征层的名字
print(layer_names)

images_per_row = 16

for layer_name, layer_activation in zip (layer_names, activations):
    n_feature = layer_activation.shape[-1] # 每层输出的特征层数
    size = layer_activation.shape[1]  #每层的特征大小
    n_cols = n_feature//images_per_row #特征图平铺的行数
    # display_grid = np.zeros((size*n_cols, images_per_row*size)) # 每层图片大小
    print('每层输出的特征层数',n_feature)
    print('每层的特征大小',size)
    print('特征图平铺的行数',n_cols)
    # print('每层图片大小',display_grid)
    print('-'*30)

    # for col in range(n_cols): #行扫描
    #     for row in  range (images_per_row): #平铺每行
    #         # print(layer_activation.shape)
    #         # print(col*images_per_row+row)
    #         channel_image = layer_activation[0,:,:,col*images_per_row+row] # 写入col*images_per_row+row特征层
    #         channel_image -= channel_image.mean() #标准化处理，增加可视化效果
    #         channel_image /= channel_image.std()
    #         channel_image *=64
    #         channel_image +=128
    #         channel_image = np.clip(channel_image, 0, 255).astype('uint8')
    #         # print(channel_image.shape)
    #         # print(display_grid[col*size:(col+1)*size, row*size:(row+1)*size].shape)
    #         display_grid[col*size:(col+1)*size, row*size:(row+1)*size] = channel_image #写入大图中
    # scale = 1./size #每组图缩放系数
    # plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
    # plt.title(layer_name)
    # plt.grid(False)
    # plt.imshow(display_grid, aspect='auto', cmap='viridis')
    # plt.show()











