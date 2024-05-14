
import os

# # GPU配置
# os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import paper_01

import matplotlib.pyplot as plt
import pandas as pd

# matplotlib支持中文设置
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams['axes.unicode_minus'] = False

'''matplotlib可视化'''

print('-'*50)
history = paper_01.history
print(history.params)   # {'verbose': 2, 'epochs': 2, 'steps': 30}
print(history.history.keys())
# dict_keys(['loss', 'sparse_categorical_accuracy',
#           'val_loss', 'val_sparse_categorical_accuracy'])

'''<1> history曲线'''
# acc = history.history['sparse_categorical_accuracy']
# val_acc = history.history['val_sparse_categorical_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs, acc)
# plt.plot(epochs, val_acc)
# plt.title('Training and Validation accuracy')
# plt.legend(['Training Accuracy', "Validation Accuracy"])
#
# plt.figure()
#
# plt.plot(epochs, loss)
# plt.plot(epochs, val_loss)
# plt.title('Training and Validation loss')
# plt.legend(['Training loss', 'Validation loss'])
#
# plt.show()

'''<2> history曲线 混合'''
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.2)
plt.xlabel("迭代次数")
plt.show()








