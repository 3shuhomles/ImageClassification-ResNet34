
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import time

# # GPU配置
# os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 0为独显，1为集显 会报错 自动跳转CPU

'''config'''
EPOCHES = 30
LEARNINGRATE = 0.000001
NAME = f'CNN_34_{EPOCHES}_{LEARNINGRATE}'
SEED = 117862


# matplotlib支持中文设置
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False

# 设置显示格式
np.set_printoptions(threshold=np.inf)

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
    image_size=(128,128),
    shuffle=True,
    seed = SEED,  # 随机种子 between 0 and 2**32-1
    subset='training',
    validation_split=0.1    # 十折交叉验证

    # shuffle=False,
    # subset='training',
    # validation_split=0.1
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(scale=1./255),

    tf.keras.layers.Conv2D(64,(7,7),strides=(2,2),activation='relu',padding='same',input_shape=(128,128,3)),
    tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2)),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64,(3,3),strides=(1,1),activation='relu',padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'),

    tf.keras.layers.Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(1,1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(1,1), activation='relu', padding='same'),

    tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same'),

    tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same'),

    tf.keras.layers.AveragePooling2D(pool_size=(2, 2),strides=1),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation='softmax')   # sigmoid1
])

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(), # BinaryCrossentropy
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNINGRATE),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)

cheskpoint_str = f"./checkpoint_{NAME}/Model_{NAME}.ckpt"
# 设置断点
checkpoint_save_path = cheskpoint_str
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only = True,
                                                 save_best_only = False,
                                                 monitor = 'val_loss',
                                                 mode = min
                                                 )

history = model.fit(
    train_BatchDataset,
    steps_per_epoch=None,
    epochs=EPOCHES,
    verbose=1,
    validation_freq=1,
    callbacks=[cp_callback]
)
# ,callbacks=[cp_callback]

model.summary()

save_str = f"./HDF5_{NAME}"
'''保存模型'''
save_path = save_str
model.save(save_path)
print('Model保存成功')

history_str = f'.\dataset\model_figure/history_{NAME}.txt'
'''保存history信息'''
history_dict = history.history
file = open(history_str, 'w')
for key,value in history_dict.items():
    # print(key,type(key),type(value))
    file.write(key + '\n')
    file.write(str(value) + '\n')
file.close()
print('history保存成功')

weight_str = f'.\dataset\model_figure\weights_{NAME}.txt'
'''保存权重'''
# print(model.trainable_variables)
strat = time.time()
file = open(weight_str, 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    if len(v.shape) == 4:
        v_numpy = np.rollaxis(v.numpy(),2,0)
        v_numpy = np.rollaxis(v_numpy,3,0)
    file.write(str(v_numpy) + '\n')
file.close()
print('权重保存成功')
print("用时：",time.time()-strat)

pltsave_str = f".\dataset\model_figure\Loss_and_CategoricalAccuracy_{NAME}.png"
'''history曲线'''
pd.DataFrame(history.history).plot(figsize=(10, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.2)
plt.xlabel("迭代次数")
plt.xticks(np.arange(1,EPOCHES+1,1))
plt.savefig(fname=pltsave_str)
print('history图像保存成功')
plt.show()


