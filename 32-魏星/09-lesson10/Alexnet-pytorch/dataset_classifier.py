import os
import re
import shutil

'''
数据集分类
'''

origin_path = './data/image/cifar-10-batches-py'
target_path_0 = './data/image/train_0/0'
target_path_1 = './data/image/train_0/1'

file_list = os.listdir(origin_path)

for i in range(len(file_list)):
    old_path = os.path.join(origin_path, file_list[i])
    print(file_list[i])
    result = re.findall(r'\w+', file_list[i])[0]
    if result == 'cat':
        shutil.copy(old_path, target_path_0)
    else:
        shutil.copy(old_path, target_path_1)
