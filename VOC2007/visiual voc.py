# -*- coding: utf-8 -*-
"""
Created on Sat May 14 10:53:32 2022

@author: GJZ
"""

import os
import cv2
import re
 
pattens = ['name','xmin','ymin','xmax','ymax']
 
def get_annotations(xml_path):
    bbox = []
    with open(xml_path,'r') as f:
        text = f.read().replace('\n','return')
        p1 = re.compile(r'(?<=<object>)(.*?)(?=</object>)')
        result = p1.findall(text)
        for obj in result:
            tmp = []
            for patten in pattens:
                p = re.compile(r'(?<=<{}>)(.*?)(?=</{}>)'.format(patten,patten))
                if patten == 'name':
                    tmp.append(p.findall(obj)[0])
                else:
                    tmp.append(int(float(p.findall(obj)[0])))
            bbox.append(tmp)
    return bbox
 
def save_viz_image(image_path,xml_path,save_path):
    bbox = get_annotations(xml_path)
    image = cv2.imread(image_path)
    for info in bbox:
        cv2.rectangle(image,(info[1],info[2]),(info[3],info[4]),(255,255,255),thickness=2)
        cv2.putText(image,info[0],(info[1],info[2]),cv2.FONT_HERSHEY_PLAIN,1.2,(255,255,255),2)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    name = os.path.join(save_path,image_path.split('\\')[-1]) # if linux ,use '/'
    cv2.imwrite(name,image)
 
    
# 图片目录不对，imshow不出图片 
    
if __name__ == '__main__':
    image_dir = 'JPEGImages'
    xml_dir = 'Annotations'
    save_dir = 'viz_images'
    image_list = os.listdir(image_dir)
    a=0
    for i in  image_list:
        image_path = os.path.join(image_dir,i)
        xml_path = os.path.join(xml_dir,i.replace('.jpg','.xml'))
        save_viz_image(image_path,xml_path,save_dir)
        a+=1
        if a == 20:
            break
    
# from gettext import find
# import os
# from xml.etree import ElementTree as ET
# import cv2
 
 
# def drawBoxOnVOC(img, xml, out, label=False):
 
#     per=ET.parse(xml)
#     image = cv2.imread(img)
#     imgName = img.split('/')[-1]
#     root = per.getroot()
 
#     p=root.findall('object')
 
#     for oneper in p:
#         # print(oneper.find('name').text)
#         bndbox = oneper.find('bndbox')
#         x1 = (int)(bndbox.find('xmin').text)
#         y1 = (int)(bndbox.find('ymin').text)
#         x2 = (int)(bndbox.find('xmax').text)
#         y2 = (int)(bndbox.find('ymax').text)
#         # 各参数依次是：图片，添加的文字，左上角坐标(整数)，字体，字体大小，颜色，字体粗细
#         # cv2.putText(img, oneper.find('name').text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#         image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
#     cv2.imwrite(os.path.join(out, imgName), image)
 
# rootPath = 'data/images'
# imgList = os.listdir(rootPath)
# for imgName in imgList:
#     print(imgName)
#     (name, ex) = os.path.splitext(imgName)
#     img = os.path.join(rootPath, imgName)
#     xml = os.path.join('data/xml', name + '.xml')
#     drawBoxOnVOC(img, xml, 'dataOut')
