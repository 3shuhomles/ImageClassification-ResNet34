# 导入包
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from keras import Model

# GPU配置
# os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 设置显示格式
np.set_printoptions(threshold=np.inf)

'''导入数据集主地址'''
base_dir = 'D:\code\paper\paper\dataset\classification'
print(tf.keras.preprocessing.image.ImageDataGenerator)
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

'''训练集'''
train_bat_dir = os.path.join(train_dir,'bat')
train_leopard_dir = os.path.join(train_dir,'leopard')
train_otter_dir = os.path.join(train_dir,'otter')

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(112,112),  # 指定resize的大小
    batch_size=900,
    class_mode='sparse'    # binary categorical
)

z = train_generator.next()
x_train = np.array(z[0],dtype=float)
y_train = np.concatenate((np.zeros(300),np.ones(300),np.ones(300)+1))

'''验证集'''
validation_bat_dir = os.path.join(train_dir,'bat')
validation_leopard_dir = os.path.join(train_dir,'leopard')
validation_otter_dir = os.path.join(train_dir,'otter')

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(112,112),  # 指定resize的大小
    batch_size=150,
    class_mode='sparse'
)
z_2 = validation_generator.next()
x_validation = np.array(z_2[0],dtype=float)
y_validation = np.concatenate((np.zeros(50),np.ones(50),np.ones(50)+1))

'''测试集'''
test_bat_dir = os.path.join(test_dir,'bat')
test_leopard_dir = os.path.join(test_dir,'leopard')
test_otter_dir = os.path.join(test_dir,'otter')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(112,112),  # 指定resize的大小
    batch_size=20,
    class_mode='sparse'    # binary categorical
)
z_3 = test_generator.next()
x_test = np.array(z_3[0],dtype=float)
y_test = np.concatenate((np.zeros(20),np.ones(20),np.ones(20)+1))



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


class ResNet18(Model):

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        # 结构定义
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
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
        self.f1 = tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


# 运行，一共4个元素，所以block执行4次，每次有2个
model = ResNet18([2, 2, 2, 2])

# 设置优化器等
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
#
# # 设置断点
# checkpoint_save_path = "./checkpoint/ResNet18.ckpt"
# if os.path.exists(checkpoint_save_path + '.index'):
#     print('-------------load the model-----------------')
#     model.load_weights(checkpoint_save_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#                                                  save_weights_only=True,
#                                                  save_best_only=True)
#
history = model.fit(x_train, y_train, batch_size=30, epochs=3, validation_data=(x_validation, y_validation)
                    ,validation_freq=1)
# # 断点续传参数
# # ,callbacks=[cp_callback]
#
# # 显示结果
model.summary()

# # 保存权重
# # print(model.trainable_variables)
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()
#
# # 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
