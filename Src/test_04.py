
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

'''绘图实验'''
# # 显示中文
# plt.rcParams["font.sans-serif"] = "SimHei"
# plt.rcParams['axes.unicode_minus'] = False
#
# x=np.linspace(-2*np.pi,2*np.pi,400)
# siny=np.sin(x)
# cosy=np.cos(x)
#
# plt.plot(x,siny,color="red",label="sin(x)")
# plt.plot(x,cosy,color="blue",label="cos(x)",linestyle="--")
# plt.xlabel("x")
# plt.ylabel("sin(x) 或者 cos(x)")
#
# plt.yticks(np.arange(-1,1,0.2))
#
# plt.title("三角函数图")
# plt.legend()
# # plt.imshow(fname=r".\dataset\model_figure\pic.png")
# # plt.savefig(fname=r".\dataset\model_figure\pic.png",figsize=[10,10])
# plt.show()


'''卷积核维度测试'''

'''设置显示格式'''
np.set_printoptions(threshold=np.inf)

# matplotlib支持中文设置
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False

'''config'''
EPOCHES = 1
LEARNINGRATE = 0.0001
NAME = f'CNN_18_TEST'
SEED = 777

'''导入数据集地址'''
base_dir = r'D:\code\paper\paper\dataset\TG2'
train_dir = os.path.join(base_dir,'train')
# validation_dir = os.path.join(base_dir,'validation')

'''训练集'''
train_BatchDataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=25,
    image_size=(64,64),
    shuffle=True,
    seed = SEED,  # 随机种子 between 0 and 2**32-1
    subset='training',
    validation_split=0.1    # 十折交叉验证
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(scale=1./255),

# 定义一扁卷积核，观察输出卷积核的高宽
    tf.keras.layers.Conv2D(32,(5,7),strides=(2,2),activation='relu',padding='same',input_shape=(64,64,3)),
    tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2)),


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation='softmax')   # sigmoid1
])

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(), # BinaryCrossentropy
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNINGRATE),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)

history = model.fit(
    train_BatchDataset,
    steps_per_epoch=None,
    epochs=EPOCHES,
    verbose=1,
    validation_freq=1
)

model.summary()

weight_str = f'.\dataset\model_figure\TEST_weights.txt'
'''维度测试'''
print(len(model.trainable_variables))
for i in model.trainable_variables:
    print(i.name,i.shape,i.numpy().shape)
    conv = i.numpy()
    break
example = conv[:,:,:,0]
print('-'*50)
print(example)
file = open(weight_str, 'w')
for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
    if len(v.shape) == 4:
        v_numpy = np.rollaxis(v.numpy(),2,0)
        v_numpy = np.rollaxis(v_numpy,3,0)
    file.write(str(v_numpy) + '\n')
file.close()
print('测试权重文件保存成功')










