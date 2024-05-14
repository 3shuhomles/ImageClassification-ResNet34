

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import time

# # GPU配置
# os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 0为独显，1为集显 会报错 自动跳转CPU

'''config'''
EPOCHES = 30
LEARNINGRATE = 0.0001
generation = '2nd'
# NAME = f'ResNet34_{EPOCHES}_{LEARNINGRATE}'
NAME = f'ResNet34_{EPOCHES}_{LEARNINGRATE}_{generation}'
# NAME = f'ResNet34_{EPOCHES}_{LEARNINGRATE}_Notshuffle'

# load_path = f"./HDF5_{NAME}"
load_path = f"E:\ResNet34\HDF5_{NAME}"
model_2 = tf.keras.models.load_model(load_path)

# matplotlib支持中文设置
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False

# 设置显示格式
np.set_printoptions(threshold=np.inf)

'''导入数据集地址'''
base_dir = r'D:\code\paper\paper\dataset\TG2'
test_dir = os.path.join(base_dir,'test')

'''测试集'''
test_BatchDataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=25,
    image_size=(128,128),
    shuffle=False,
)

'''evaluate'''
y_evaluate = model_2.evaluate(
    test_BatchDataset,
    verbose = 1,
    steps = None
)
print("y_evaluate：",y_evaluate)
print(type(y_evaluate))
# loss:CategoricalCrossentropy,accuracy:CategoricalAccuracy
print('-'*50)
'''保存测试集损失与精确度'''
file = open(fr'E:\ResNet34\New_folder\test_result\PredictResult_{NAME}.txt', 'w')
for v in y_evaluate:
    file.write(str(v) + '\n')
file.write('\n')
file.close()



'''predict'''
y_predict = model_2.predict(
    test_BatchDataset,
    verbose = 1,
    steps = None
)
# print("y_predict：",y_predict)
print(type(y_predict))
print(y_predict.shape)
# print(y_predict[0:100,:].shape)
'''保存测试集各图像softmax结果'''
file = open(fr'E:\ResNet34\New_folder\test_result\PredictResult_{NAME}.txt', 'a')
file.write('以下为各图像softmax函数结果：' + '\n')
for v in y_predict:
    file.write(str(v) + '\n')
    # print(str(v) + '\n')
file.write('\n')
file.close()
'''保存测试集各图像softmax结果按最大值索引降序排列数组'''
file = open(fr'E:\ResNet34\New_folder\test_result\PredictResult_{NAME}.txt', 'a')
file.write('以下为各图像softmax函数结果按最大值索引降序排列列表：' + '\n')
for v in y_predict:
    file.write(str(np.argsort(v)[::-1]) + '\n')     #np数组
    # print(str(np.argsort(v)[::-1]) + '\n')
file.write('\n')
file.close()

'''保存图像所属类名'''
result_lst = []
class_name = test_BatchDataset.class_names
for i in y_predict:
    # print(i,type(i))
    lst = np.argsort(i)[::-1]
    # print(lst)  # 最大值索引降序排序
    classification_result = class_name[lst[0]]
    result_lst.append(classification_result)
# print(result_lst)
# print(len(result_lst))
file = open(fr'E:\ResNet34\New_folder\test_result\PredictResult_{NAME}.txt', 'a')
file.write('以下为各图像分类结果：' + '\n')
for i in result_lst:
    file.write(i + '\n')
file.write('\n')
file.close()

print('测试结果保存成功')




'''从训练好的模型加载权重信息'''
# weight_str = f'E:\ResNet34\weights_{NAME}.txt'
# '''保存权重'''
# # print(model.trainable_variables)
# start = time.time()
# file = open(weight_str, 'w')
# for v in model_2.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     if len(v.shape) == 4:
#         v_numpy = np.rollaxis(v.numpy(),2,0)
#         v_numpy = np.rollaxis(v_numpy,3,0)
#         file.write(str(v_numpy) + '\n')
#     else:
#         file.write(str(v.numpy()) + '\n')
# file.close()
# print('权重保存成功')
# print('用时：',time.time()-start)
# print('\n')
# for W in model_2.trainable_variables:
#     print(W.name,W.shape)
# print('\n')