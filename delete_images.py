# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:55:20 2022

@author: GJZ
"""

import os
import sys

data = []
for line in open("VOCdevkit_person/VOC2012/ImageSets/Main/train.txt", "r"):  # txt文件，里面包含需要删除的图片名称
    data.append(line[:-1])

picturedir="VOCdevkit_person/VOC2012/JPEGImages"  #图片路径
anndir="VOCdevkit_person/VOC2012/Annotations" # 标签路径

filelist1=os.listdir(picturedir)


for f in filelist1:
    if f [:-4] not in data:
        del_file = picturedir + '/' + f
        os.remove(del_file)  # 删除文件
        print("delete  ", del_file)

filelist2=os.listdir(anndir)

for f in filelist2:
    if f [:-4] not in data:
        del_file = anndir + '/' + f
        os.remove(del_file)  # 删除文件
        print("delete  ", del_file)