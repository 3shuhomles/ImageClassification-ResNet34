
from inspect import isfunction
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

'''查看方法是否存在'''
# print(isfunction(tf.keras.Model.fit))
# print(isfunction(tf.keras.Model.fit_generator))

# print(list(range(0,1000,100)))


# nu = np.ones((1,100))
# print(nu.shape)
# print(np.shape([[1],[2]]))
'''数组按行输出'''
# num = np.ones((1000,10))
# for i in range(0,1000):
#     print(num[i,:])

'''顺序读取文件夹图片'''
# import pathlib
# import matplotlib.pyplot as plt
# data_dir = r'D:\code\paper\paper\dataset\TG2\test'
# data_dir = pathlib.Path(data_dir)
# image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)
# one = list(data_dir.glob('*/*.jpg'))
# image = plt.imread(str(one[100]))
# plt.imshow(image)
# plt.show()

'''查看tensorflow读取的图片转化为数组的维度'''
# 导入数据集地址
base_dir = r'D:\code\paper\paper\dataset\TG2'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

# 训练集ImageDataset
train_BatchDataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=25,
    image_size=(128,128),
    shuffle=False,
    # seed = 111,  # 随机种子 between 0 and 2**32-1
    subset='training',
    validation_split=0.1    # 十折交叉验证
)

'''dataset类别标签'''
print(train_BatchDataset.class_names)
for k,v in train_BatchDataset.take(1):
    images = k
    labels = v
print(images.shape,type(images))

'''图像可视化'''
# picture1 = images[0,:,:,:].numpy()
# plt.imshow(picture1/255)
# plt.show()
# print(labels.shape,type(labels))

'''读取One-Hot标签对应的label'''
# print(train_BatchDataset.class_names[labels[0]])
# print(labels[0].numpy().astype(int))
lst = labels[0].numpy().astype(int).tolist()
# print(lst)
print(train_BatchDataset.class_names[lst.index(max(lst))])












