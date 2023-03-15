# 处理数据集

# 原数据集
# HANDS
# --Scripts

# --Subject1
#   --Subject1
#   --Subject1_Depth
#   --Subject1.mat
#   --Subject1.txt

# --...
# --Subject5

# 修改为yolo格式，其中train:val = 9:1
# HANDS
# --Subject1
#   --images_color
#   --images_depth
#   --labels
# --...
# --Subject5
# --train_color.txt
# --train_depth.txt
# --val_color.txt
# --val_depth.txt
import os
from pathlib import Path
import pandas as pd
import numpy as np
import random
import cv2 as cv
import my_utils


def make_dir(path):
    if path.is_dir():
        path.mkdir(parents=True)


parent = Path.cwd().parent
dp = parent / 'HANDS'  # dataset_path
if not dp.is_dir():
    exit(0)

mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 5, 12: 6, 13: 6, 14: 7, 15: 7, 16: 8,
           17: 8, 18: 9, 19: 9, 20: 10, 21: 10, 22: 11, 23: 12, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15}

none = '[0 0 0 0]'  # 0

# png的文件路径总列表
color = []
# depth = []
for i in range(1, 6):
    sp = dp / f'Subject{i}' #Subjecti path
    # # 重命名文件夹
    # if (sp / f'Subject{i}').is_dir():
    #     (sp / f'Subject{i}').rename(sp / 'images_color')
    # if (sp / f'Subject{i}_Depth').is_dir():
    #     (sp / f'Subject{i}_Depth').rename(sp / 'images_depth')
    # # 重命名文件
    # for f in (sp / 'images_color').glob('*.png'):
    #     f.rename(f.parent / (f.name.partition('_')[0] + '.png'))
    # for f in (sp / 'images_depth').glob('*.png'):
    #     f.rename(f.parent / (f.name.partition('_')[0] + '.png'))
    # 收集文件路径
    # color += list((sp / 'images_color').glob('*.png'))
    # depth += list((sp / 'images_depth').glob('*.png'))
    # mkdir labels
    labels = sp / 'labels'
    if not labels.is_dir():
        labels.mkdir()
    # make label txts
    txt = sp / f'Subject{i}.txt'
    table = pd.read_csv(txt, sep=',')
    for _, line in table.iterrows():
        number = line[0].rpartition(os.sep)[-1].partition('_')[0]  # 找到图片序号
        # 创建number.txt文件
        with open(labels / (number + '.txt'), 'w') as label:
            for k, v in enumerate(line[2:]):
                # not [0 0 0 0]
                if v != none:
                    xywh = v.strip('[').rstrip(']').split(' ')
                    xywh = np.array(xywh, dtype=float)
                    xywh[0] = xywh[0] + xywh[2]/2
                    xywh[1] = xywh[1] + xywh[3]/2
                    xywh = xywh / [960, 540, 960, 540]
                    label.write(f'{mapping[k]} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n')  # space分格（类别 ，xc，yc，w，h）归一化
#
# # 划分 9:1
# val = random.sample(color, len(color) // 10)
# train = set(color) - set(val)
# train = list(train)
# # 字典序
# train.sort()
# val.sort()
#
# # write into file
# with open(dp / 'train_color.txt', 'w') as f:
#     for path in train:
#         f.write('./' + str(path.relative_to(dp)).replace('\\', '/') + '\n')  # relative to HANDS
# with open(dp / 'train_depth.txt', 'w') as f:
#     for path in train:
#         f.write('./' + str(path.relative_to(dp)).replace('\\', '/').replace('color', 'depth') + '\n')
# with open(dp / 'val_color.txt', 'w') as f:
#     for path in val:
#         f.write('./' + str(path.relative_to(dp)).replace('\\', '/') + '\n')  # relative to HANDS
# with open(dp / 'val_depth.txt', 'w') as f:
#     for path in val:
#         f.write('./' + str(path.relative_to(dp)).replace('\\', '/').replace('color', 'depth') + '\n')


# # 将深度图转化为不失精度的rgb图，相当于在各自的空间上做了一个映射
# for i in range(1, 6):
#     sp = dp / f'Subject{i}'  # Subjecti path
#     if not (sp / 'images_depth_t').is_dir():
#         (sp / 'images_depth_t').mkdir()
#     for f in (sp / 'images_depth').glob('*.png'):
#         img = my_utils.depth_to_pseudo_color(cv.imread(str(f), cv.IMREAD_UNCHANGED))
#         cv.imwrite(str(sp / 'images_depth_t' / f.name), img)

# 修改文件中的路径
# with open(dp / 'train_depth.txt', 'w') as f:
#     for path in train:
#         f.write('./' + str(path.relative_to(dp)).replace('\\', '/').replace('color', 'depth') + '\n')

# try:
#     (dp / 'images_color/train').mkdir(parents=True)
#     (dp / 'images_color/val').mkdir(parents=True)
#     (dp / 'images_depth/train').mkdir(parents=True)
#     (dp / 'images_depth/val').mkdir(parents=True)
#     (dp / 'labels/train').mkdir(parents=True)
#     (dp / 'labels/val').mkdir(parents=True)
# except:
#     pass


# # 移动文件
# for n in range(1, 6):
#     p = dp / ('Subject' + str(n))
#     s = p / ('Subject' + str(n))
#     s_d = p / ('Subject' + str(n) + '_Depth')
#     # 移动彩色图
#     for f in s.iterdir():
#         target = (dp / 'images_color') / (f.name.partition('_') + '.png')
#         f.replace(target)
#     # 移动深度图
#     for f in s_d.iterdir():
#         target = (dp / 'images_depth') / (f.name.partition('_') + '.png')
#         f.replace(target)
#     # 移动txt
#     txt = list(p.glob('*.txt'))[0]
#     txt.replace(dp / 'labels' / txt.name)

# # 重命名文件
# for f in (dp / 'images_color').glob('*.png'):
#     f.rename(f.parent / (f.name.partition('_')[0] + '.png'))
# for f in (dp / 'images_depth').glob('*.png'):
#     f.rename(f.parent / (f.name.partition('_')[0] + '.png'))


# # classes_list
# classes = pd.read_csv(dp / 'Subject1/Subject1.txt', sep=',', nrows=1, header=None)
# classes = classes.loc[:, 2:]
# classes = [str(classes[i + 2][0]) for i in range(len(classes.columns))]
#
# print(classes)
# # ['Punch_VFR', 'Punch_VFL', 'One_VFR', 'One_VFL', 'Two_VFR', 'Two_VFL', 'Three_VFR', 'Three_VFL',
# #  'Four_VFR', 'Four_VFL', 'Five_VFR', 'Five_VFL', 'Six_VFR', 'Six_VFL', 'Seven_VFR', 'Seven_VFL', 'Eight_VFR', 'Eight_VFL',
# #  'Nine_VFR', 'Nine_VFL', 'Span_VFR', 'Span_VFL', 'Horiz_HBL', 'Horiz_HFL', 'Horiz_HBR', 'Horiz_HFR', 'Collab', 'XSign', 'TimeOut']
# nc = len(classes)
# print(nc)

['Punch_VF', 'One_VF', 'Two_VF', 'Three_VF', 'Four_VF', 'Five_VF', 'Six_VF', 'Seven_VF', 'Eight_VF', 'Nine_VF',
 'Span_VF', 'Horiz_HB', 'Horiz_HF', 'Collab', 'XSign', 'TimeOut']

mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 5, 12: 6, 13: 6, 14: 7, 15: 7, 16: 8,
           17: 8, 18: 9, 19: 9, 20: 10, 21: 10, 22: 11, 23: 12, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15}



# none = '[0 0 0 0]'
# # classes
#
# for f in (dp / 'labels').glob('Subject*.txt'):
#     table = pd.read_csv(f, sep=',')
#     for _, line in table.iterrows():
#         number = line[0].rpartition(os.sep)[-1].partition('_')[0]  # 找到图片序号
#         # 创建number.txt文件
#         with open(f.parent / (number + '.txt'), 'w') as label:
#             for k, v in enumerate(line[2:]):
#                 # not [0 0 0 0]
#                 if v != none:
#                     xywh = v.strip('[').rstrip(']').split(' ')
#                     xywh = np.array(xywh, dtype=float)
#                     xywh = xywh / [960, 540, 960, 540]
#                     label.write(f'{k + 2} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n')  # space分格（类别 ，xc，yc，w，h）归一化
# # move outside
# for f in (dp / 'labels').glob('Subject*.txt'):
#     f.replace(dp / f.name)
