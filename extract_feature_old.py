# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 18:34:07 2022

@author: GJZ
"""
import torch
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.mobilenetv1_ssd_lite_025extra import create_mobilenetv1_ssd_lite_025extra
import cv2
from vision.ssd.data_preprocessing import PredictionTransform

model=create_mobilenetv1_ssd_lite_025extra(num_classes=2, width_mult=0.25, is_test=True)  
# print (model.classification_headers[2][1])
class_names = ['Person','Background']
image_path = 'VOC2007/JPEGImages/IMG00457.jpg'
model_path = 'models/ws0.25tmax400extra0.25/mb1-ssd-lite-025extra-Epoch-399-Loss-3.2293308803013394.pth'
# load pretrained model
model.load_state_dict(torch.load(model_path))
model.cuda()
predictor = create_mobilenetv1_ssd_lite_predictor(model, candidate_size=200)

#先获取 module name ，用于在后面层中使用
for name, module in model.named_modules():
    print('==',name,
                           # '|||||||',module
                        )

# 存储中间层的 feature
total_feat_out = []
total_feat_in = []
total_module = []

# 定义 forward hook function
def hook_fn_forward(module, input, output):
    # print(module) # 用于区分模块
    # print('input', input) # 首先打印出来
    # print('output', output)
    total_module.append(module) # 然后分别存入全局 list 中
    total_feat_out.append(output) 
    total_feat_in.append(input)

names=[]
modules = model.named_modules() #
for name, module in modules:
    if name == 'extras.1.1' :
      module.register_forward_hook(hook_fn_forward)
      names.append(name)
      # module.register_backward_hook(hook_fn_backward)

# （第一维是 batch size）。

# use rand tensor as input 
input=torch.rand([ 1 , 3, 300, 300]).requires_grad_().to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))   
# use images as input
orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
transform = PredictionTransform(size=300, mean=0.0, std=1.0)
new_image = transform(image)
# boxes, labels, probs = predictor.predict(image, 10, 0.35)

with torch.no_grad():
  # out = model(input)
  outputs = predictor.predict(image, 10, 0.35)

print('==========Saved inputs and outputs==========')
for idx in range(len(total_feat_in)):
    a=total_feat_in[idx][0].reshape(-1).tolist()
    print('module: ', total_module[idx])
    print('input: ', total_feat_in[idx][0].shape,'\n'
          #,total_feat_in[idx][0]
          )
    print('output: ', total_feat_out[idx].shape,'\n'
          #,total_feat_out[idx][0]
          )
result=[names[0],' ',total_module[idx],'\n input shape = ',total_feat_in[idx][0].shape,'\n input = ' ,total_feat_in[idx][0].reshape(-1).tolist(),'\n output shape = ',total_feat_out[idx].shape, '\n output = ' ,total_feat_out[idx].reshape(-1).tolist()]
f = open('extras.1.1 feature map.txt','w')
for i in result:
    f.write(str(i))
f.close()
