# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 08:36:18 2022

@author: GJZ
"""

# B,C,H,W -  Batch, Channel, Height, Width

from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.mobilenetv1_ssd_lite_025extra import create_mobilenetv1_ssd_lite_025extra
from vision.ssd.mobilenetv1_ssd_lite_224 import create_mobilenetv1_ssd_lite_025extra_224, create_mobilenetv1_ssd_lite_predictor224
from vision.ssd.mobilenetv1_ssd_lite_277kb import create_mobilenetv1_ssd_lite_277, create_mobilenetv1_ssd_lite_predictor_277



import cv2
import torch

class_names = ['Person','Background']
# model_path = 'models/ws0.25tmax400extra0.25/mb1-ssd-lite-025extra-Epoch-399-Loss-3.2293308803013394.pth'
model_path = 'models/277/Epoch-1950-Loss-3.4940130710601807.pth'
image_path = 'image/01_3PTeam.png'

# model = create_mobilenetv1_ssd_lite_025extra(len(class_names),  width_mult=0.25 ,is_test=True)
model = create_mobilenetv1_ssd_lite_277(len(class_names),  width_mult=1.0 ,is_test=True)

# predictor = create_mobilenetv1_ssd_lite_predictor224(model, candidate_size=200)
predictor = create_mobilenetv1_ssd_lite_predictor_277(model, candidate_size=200)

model= torch.load(model_path)
# model.load(model_path)
# model = model.state_dict()

# orig_image = cv2.imread(image_path)
# image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
# boxes, labels, probs = predictor.predict(image, 10, 0.35)


all,all_layer_index,layer_name,all_pos=[],[],[],[0]
weight,weight_each_index,weight_layer_index,weight_layer_name,weight_pos=[],[],[],[],[0]
bias,bias_layer_index,bias_layer_name,bias_pos=[],[],[],[0]
batchnorm,batchnorm_each_name,batchnorm_each_index,batchnorm_pos=[],[],[],[0]

# backbone卷积后无偏置bias，因为后面是batchnorm；而后面的两层因为无batchnorm，所以有bias
for key in model: # model字典遍历 key名
    # if key=='base_net.0.0.weight':
    print(key,'\t',model[key].shape)
    #  list "all" :全部权值
    if model[key].dim():
        list = model[key].reshape(-1).tolist()
        # list "weight" :只有卷积层的权值
        if model[key].dim() == 4 :
            each_index = [model[key].shape[0],model[key].shape[1],model[key].shape[2],model[key].shape[3]]
                        # ouput,input,h,w
            weight += list
            weight_each_index += each_index
            weight_layer_index.append(len(list))
            weight_layer_name.append(key)
            weight_pos.append(weight_pos[-1]+len(list))
        # list "bias" : 只有extra后的bias的权值
        if "base_net" not in key and  "bias" in key:
            bias += list
            bias_layer_index.append(len(list))
            bias_layer_name.append(key)
            bias_pos.append(bias_pos[-1]+len(list))
        # list "batchnorm"： 顺序：gamma(weight)，beta(bias)，mean，variance
        if "base_net" in key and (".1.weight" in key or ".4.weight" in key):
            batchnorm += list
            batchnorm_each_index.append(len(list))
            batchnorm_each_name.append(key)
            batchnorm_pos.append(batchnorm_pos[-1]+len(list))
        elif "base_net" in key and "bias" in key :
            batchnorm += list
            batchnorm_each_index.append(len(list))
            batchnorm_each_name.append(key)
            batchnorm_pos[-1]=(batchnorm_pos[-1]+len(list))
        elif "running_mean" in key:
            batchnorm += list
            batchnorm_each_index.append(len(list))
            batchnorm_each_name.append(key)
            batchnorm_pos[-1]=(batchnorm_pos[-1]+len(list))
        elif "running_var" in key:
            batchnorm += list
            batchnorm_each_index.append(len(list))
            batchnorm_each_name.append(key)
            batchnorm_pos[-1]=(batchnorm_pos[-1]+len(list))
        
        all += list
        layer_name.append(key)
        all_layer_index.append(len(list))
        all_pos.append(all_pos[-1]+len(list))

result0=['weight_layer_name = ',str(weight_layer_name),'\n','weight_data = ',str(weight),'\n','weight_shape = ',str(weight_each_index),'\n', 'weight_pos = ', str(weight_pos), '\n\n']
result1=['bias_layer_name = ',str(bias_layer_name),'\n','bias_data = ',str(bias),'\n','bias_layer_shape = ',str(bias_layer_index),'\n', 'bias_pos = ', str(bias_pos), '\n\n']
result2=['batchnorm_each_name = ',str(batchnorm_each_name),'\n','batchnorm_data = ',str(batchnorm),'\n','batchnorm_each_shape = ',str(batchnorm_each_index),'\n', 'batchnorm_pos = ', str(batchnorm_pos), '\n\n']
result=result0+result1+result2
f = open('weights_277kb.txt','w')
for i in result:
    f.write(str(i))
f.close()


    

        
    