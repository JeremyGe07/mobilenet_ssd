from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.mobilenetv1_ssd_lite_025extra import create_mobilenetv1_ssd_lite_025extra
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.utils.misc import Timer
import cv2
import sys
from vision.ssd.mobilenetv1_ssd_lite_224 import create_mobilenetv1_ssd_lite_025extra_224, create_mobilenetv1_ssd_lite_predictor224
from vision.ssd.mobilenetv1_ssd_lite_025extra_4box import create_mobilenetv1_ssd_lite_025extra_4box, create_mobilenetv1_ssd_lite_predictor4box
from vision.ssd.mobilenetv1_ssd_lite_025extra_384 import create_mobilenetv1_ssd_lite_025extra384, create_mobilenetv1_ssd_lite_predictor384
from vision.ssd.mobilenetv1_ssd_lite_025extra_3box import create_mobilenetv1_ssd_lite_025extra_3box, create_mobilenetv1_ssd_lite_predictor_3box
from vision.ssd.mobilenetv1_ssd_lite_277kb import create_mobilenetv1_ssd_lite_277, create_mobilenetv1_ssd_lite_predictor_277
from vision.ssd.mobilenetv1_ssd_lite_025extra_4box_same import create_mobilenetv1_ssd_lite_025extra_4box_same


if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> [video file]')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)


if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
# elif net_type == 'mb1-ssd-lite':
#     net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
     net = create_mobilenetv1_ssd_lite(len(class_names), width_mult=1.0, is_test=True)
elif net_type == 'mb1-ssd-lite-025extra':
    net = create_mobilenetv1_ssd_lite_025extra(len(class_names),  width_mult=0.15 ,is_test=True)
elif net_type == 'mb1-ssd-lite-025extra-224':
     net = create_mobilenetv1_ssd_lite_025extra_224(len(class_names), width_mult=0.25, is_test=True)
elif net_type == 'mb1-ssd-lite-025extra-4box':
    net = create_mobilenetv1_ssd_lite_025extra_4box(len(class_names),  width_mult=0.25 ,is_test=True)
elif net_type == 'mb1-ssd-lite-025extra-4box-same':
    net = create_mobilenetv1_ssd_lite_025extra_4box_same(len(class_names),  width_mult=0.25 ,is_test=True)
elif net_type == 'mb1-ssd-lite-025extra-384':
     net = create_mobilenetv1_ssd_lite_025extra384(len(class_names), width_mult=0.25, is_test=True)
elif net_type == 'mb1-ssd-lite-025extra-3box':
    net = create_mobilenetv1_ssd_lite_025extra_3box(len(class_names),  width_mult=0.25 ,is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb3-large-ssd-lite':
    net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb3-small-ssd-lite':
    net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite-277':
    net = create_mobilenetv1_ssd_lite_277(len(class_names),  width_mult=1.0 ,is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite' or net_type == 'mb1-ssd-lite-025extra':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200,nms_method='hard')
elif net_type == 'mb1-ssd-lite-025extra-224':
    predictor = create_mobilenetv1_ssd_lite_predictor224(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite-025extra-4box' or net_type ==  'mb1-ssd-lite-025extra-4box-same' :
    predictor = create_mobilenetv1_ssd_lite_predictor4box(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite-025extra-384':
    predictor = create_mobilenetv1_ssd_lite_predictor384(net, candidate_size=200,nms_method='hard')
elif net_type == 'mb1-ssd-lite-025extra-3box':
    predictor = create_mobilenetv1_ssd_lite_predictor_3box(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite' or net_type == "mb3-large-ssd-lite" or net_type == "mb3-small-ssd-lite":
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite-277':
    predictor = create_mobilenetv1_ssd_lite_predictor_277(net, candidate_size=200)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # floatq
fourcc = cv2.VideoWriter_fourcc(*'XVID')#视频编码格式
out = cv2.VideoWriter('4box_me.avi',fourcc,fps,(int(width),int(height)))#第三个参数为帧率，第四个参数为每帧大小
# baseResultdiouHard0.23num13candi200   base+sepConvResultdiouHard0.23num13candi200
timer = Timer()
while True:
    ret, orig_image = cap.read()
    
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 13, 0.35) # 这个改的是probs score，不是nms的iou，nms的iou在predictor的第十行改。
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (int(box[0])+20, int(box[1])+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    out.write(orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
