
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from keras import Model
from keras.preprocessing.image import ImageDataGenerator

# matplotlib支持中文设置
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False

# 设置显示格式(数组过长，输出到txt会用......省略)
np.set_printoptions(threshold=np.inf)
# 不以科学计数法输出
# np.set_printoptions(suppress=True)

'''config'''
EPOCHES = 5
LEARNINGRATE = 0.001
# 0.001 0.0001 0.00001
# generation = '3rd'
NAME = f'ResNet50_{EPOCHES}_{LEARNINGRATE}'
# NAME = f'ResNet34_{EPOCHES}_{LEARNINGRATE}_{generation}'
SEED = 1111116

model = tf.keras.applications.resnet50.ResNet50(
    weights=None,
    classes=10
)

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(), # BinaryCrossentropy
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNINGRATE),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)

model.summary()

'''导入数据集地址'''
base_dir = r'D:\code\paper\paper\dataset\TG2'
train_dir = os.path.join(base_dir,'train')
# validation_dir = os.path.join(base_dir,'validation')
# test_dir = os.path.join(base_dir,'test')


'''训练集'''
train_BatchDataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=5,
    image_size=(224,224),
    # shuffle=True,
    seed = SEED,  # 随机种子 between 0 and 2**32-1
    shuffle=True,
    subset='training',
    validation_split=0.1    # 十折交叉验证

    # shuffle=False,
    # subset='training',
    # validation_split=0.1
)


history = model.fit(
    train_BatchDataset,
    steps_per_epoch=None,
    epochs=EPOCHES,
    verbose=1,
    validation_freq=1,
    workers=4
)

save_str = f"./HDF5_{NAME}"
'''保存模型'''
save_path = save_str
model.save(save_path)
print('Model保存成功')

# pltsave_str = f".\dataset\model_figure\Loss_and_CategoricalAccuracy_{NAME}.png"
'''history曲线'''
pd.DataFrame(history.history).plot(figsize=(10, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.2)
plt.xlabel("迭代次数")
plt.xticks(np.arange(1,EPOCHES+1,1))
# plt.savefig(fname=pltsave_str)
# print('history图像保存成功')
plt.show()













