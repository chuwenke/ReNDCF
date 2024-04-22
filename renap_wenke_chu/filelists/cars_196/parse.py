import scipy.io
import os
import re
import shutil

source = './'
target = './'
data = scipy.io.loadmat('./cars_annos.mat')
class_names = data['class_names']
annotations = data['annotations']
# print(annotations)

for i in range(annotations.shape[1]):
    name = str(annotations[0, i][0])[2:-2]  # 提取出图片名字
    image_path = os.path.join(source, name)
    print(image_path)
    clas = int(annotations[0, i][5])  # 提取出图片的类别属性
    class_name = str(class_names[0, clas - 1][0]).replace(' ', '_')
    class_name = class_name.replace('/', '')  # 避免路径出错
    target_path = os.path.join(target, class_name)
    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    print(target_path)
    shutil.copy(image_path, target_path)

# print(class_names.shape)
# print(class_names[0, 0][0])
# print(class_names[0, 0])
#
#
# for j in range(class_names.shape[1]):
#     class_name = str(class_names[0, j][0]).replace(' ', '_')
#     print(class_name)
