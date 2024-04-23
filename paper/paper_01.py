
import os
# GPU配置
# os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
# print('GPU配置：',tf.test.is_gpu_available())  # 检查GPU是否配置成功

# from tensorflow.keras.optimizers import Adam
        # tf.keras.optimizers.Adam
from keras.preprocessing.image import ImageDataGenerator
        # tf.keras.preprocessing.image.ImageDataGenerator

# print(tf.__version__)

base_dir = './dataset/classification'
print(tf.keras.preprocessing.image.ImageDataGenerator)
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')

# 训练集
train_bat_dir = os.path.join(train_dir,'bat')
train_leopard_dir = os.path.join(train_dir,'leopard')
train_otter_dir = os.path.join(train_dir,'otter')

# 验证集
validation_bat_dir = os.path.join(train_dir,'bat')
validation_leopard_dir = os.path.join(train_dir,'leopard')
validation_otter_dir = os.path.join(train_dir,'otter')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(112,112,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dense(3,activation='softmax')   # sigmoid1
])


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(300,activation="relu"),
#     tf.keras.layers.Dense(100, activation="relu"),
#     tf.keras.layers.Dense(3, activation="softmax"),
# ])



# model.compile(loss="sparse_categorical_crossentropy",
#               optimizer="sgd",
#               metrics=["accuracy"]
# )

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), # BinaryCrossentropy
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

model.summary()

'''数据预处理'''
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(112,112),  # 指定resize的大小
    batch_size=30,
    class_mode='sparse'    # binary categorical
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(112,112),  # 指定resize的大小
    batch_size=30,
    class_mode='sparse'
)

from inspect import isfunction
print(isfunction(tf.keras.models.Sequential.fit_generator))

'''tensorboard展示'''
# tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./tensorboard_logs")

history = model.fit(
    train_generator,
    steps_per_epoch=30,
    epochs=3,
    validation_data=validation_generator,
    validation_steps=5,
    verbose=2,
    # callbacks=[tf_callback]
)
# print(history)








