
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from keras import Model
from keras.preprocessing.image import ImageDataGenerator

# GPU配置
# os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 0为独显，1为集显 会报错 自动跳转CPU

'''config'''
EPOCHES = 1
LEARNINGRATE = 0.000001
# 0.001 0.0001 0.00001
# generation = '2nd'
NAME = f'ResNet34_{EPOCHES}_{LEARNINGRATE}_Notshuffle'
# NAME = f'ResNet34_{EPOCHES}_{LEARNINGRATE}_{generation}_Notshuffle'


# matplotlib支持中文设置
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False

# 设置显示格式(数组过长，输出到txt会用......省略)
np.set_printoptions(threshold=np.inf)
# 不以科学计数法输出
# np.set_printoptions(suppress=True)

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
    batch_size=16,
    image_size=(128,128),
    shuffle=False,
    # seed = SEED,  # 随机种子 between 0 and 2**32-1
    subset='training',
    validation_split=0.1    # 十折交叉验证

    # shuffle=False,
    # subset='training',
    # validation_split=0.1
)
# 若指定validation_split，则需指定subset，意为，划分出的一部分数据是原始的子集
#   subset='training'表示，划分出的数据是属于训练集的一部分，划分出validation_split用作验证集
# 若指定validation，则需shuffle='False'，因为tenso内部validation_split在shuffle之前，
#   会导致验证集样本不均匀


# 构建ResNetBlock的class
class ResnetBlock(Model):
    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        # 第1个部分
        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        # 第2个部分
        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        # residual等于输入值本身，即residual=x
        residual = inputs
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        # 如果维度不同则调用代码，否则不执行
        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        out = self.a2(y + residual)
        return out


class ResNet34(Model):

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet34, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        # 结构定义
        self.c1 = Conv2D(self.out_filters, (7,7), strides=2, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


# 运行，一共4个元素，所以block执行4次，每次分别有3,4,6,3个
model = ResNet34([3, 4, 6, 3])

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
    workers=4,
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
file = open(weight_str, 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    if len(v.shape) == 4:
        v_numpy = np.rollaxis(v.numpy(),2,0)
        v_numpy = np.rollaxis(v_numpy,3,0)
        file.write(str(v_numpy) + '\n')
    else:
        file.write(str(v.numpy()) + '\n')
file.close()
print('权重保存成功')
print('\n')
for W in model.trainable_variables:
    print(W.name,W.shape)
print('\n')

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


