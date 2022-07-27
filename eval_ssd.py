import torch
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite_025extra import create_mobilenetv1_ssd_lite_025extra
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.mobilenetv1_ssd_lite_224 import create_mobilenetv1_ssd_lite_025extra_224, create_mobilenetv1_ssd_lite_predictor224
from vision.ssd.mobilenetv1_ssd_lite_025extra_4box import create_mobilenetv1_ssd_lite_025extra_4box, create_mobilenetv1_ssd_lite_predictor4box
from vision.ssd.mobilenetv1_ssd_lite_025extra_384 import create_mobilenetv1_ssd_lite_025extra384, create_mobilenetv1_ssd_lite_predictor384
from vision.ssd.mobilenetv1_ssd_lite_025extra_3box import create_mobilenetv1_ssd_lite_025extra_3box, create_mobilenetv1_ssd_lite_predictor_3box
from vision.ssd.mobilenetv1_ssd_lite_277kb import create_mobilenetv1_ssd_lite_277, create_mobilenetv1_ssd_lite_predictor_277
from vision.ssd.mobilenetv1_ssd_lite_5693kb import create_mobilenetv1_ssd_lite_5693, create_mobilenetv1_ssd_lite_predictor_5693
from vision.ssd.mobilenetv1_ssd_lite_1679kb import create_mobilenetv1_ssd_lite_1679, create_mobilenetv1_ssd_lite_predictor_1679
from vision.ssd.mobilenetv1_ssd_lite_398kb import create_mobilenetv1_ssd_lite_398, create_mobilenetv1_ssd_lite_predictor_398
from vision.ssd.mobilenetv1_ssd_lite_3184kb import create_mobilenetv1_ssd_lite_3184, create_mobilenetv1_ssd_lite_predictor_3184
from vision.ssd.mobilenetv1_ssd_lite_830kb import create_mobilenetv1_ssd_lite_830, create_mobilenetv1_ssd_lite_predictor_830
from vision.ssd.mobilenetv1_ssd_lite_2197kb import create_mobilenetv1_ssd_lite_2197, create_mobilenetv1_ssd_lite_predictor_2197
from vision.ssd.mobilenetv1_ssd_lite_303kb import create_mobilenetv1_ssd_lite_303
from vision.ssd.mobilenet_v2_ssd_lite_v2 import create_mobilenetv2_ssd_lite_v2
from vision.ssd.mobilenetv1_ssd_lite_025extra_4box_same import create_mobilenetv1_ssd_lite_025extra_4box_same
from vision.ssd.mobilenetv1_ssd_lite_4box import create_mobilenetv1_ssd_lite_4box



# new ↓
# python eval_ssd.py --net mb1-ssd-lite-025extra  --dataset VOC2007/ --label_file models/voc-model-labels.txt --width_mult 0.25 --trained_model models/prune277ep2000/Epoch-1660-Loss-3.4978534153529575.pth 

#python eval_ssd.py --net mb1-ssd-lite-025extra  --dataset VOC2007my/ --trained_model models/ws0.25tmax400extra0.25/mb1-ssd-lite-025extra-Epoch-395-Loss-3.2181860378810336.pth --label_file models/voc-model-labels.txt --width 0.25

parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd-lite-025extra,mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
parser.add_argument('--width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV1')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    if args.dataset_type == "voc":
        dataset = VOCDataset(args.dataset,is_test=True)
    elif args.dataset_type == 'open_images':
        dataset = OpenImagesDataset(args.dataset, dataset_type="test")

    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names),  width_mult=args.width_mult ,is_test=True)
    elif args.net == 'mb1-ssd-lite-025extra':
        net = create_mobilenetv1_ssd_lite_025extra(len(class_names),  width_mult=args.width_mult ,is_test=True)
    elif args.net == 'mb1-ssd-lite-277':
        net = create_mobilenetv1_ssd_lite_277(len(class_names),  width_mult=1.0 ,is_test=True)
    elif args.net == 'mb1-ssd-lite-1679':
        net = create_mobilenetv1_ssd_lite_1679(len(class_names),  width_mult=1.0 ,is_test=True)
    elif args.net == 'mb1-ssd-lite-398':
        net = create_mobilenetv1_ssd_lite_398(len(class_names),  width_mult=1.0 ,is_test=True)
    elif args.net == 'mb1-ssd-lite-830':
        net = create_mobilenetv1_ssd_lite_830(len(class_names),  width_mult=1.0 ,is_test=True)
    elif args.net == 'mb1-ssd-lite-303':
        net = create_mobilenetv1_ssd_lite_303(len(class_names),  width_mult=1.0 ,is_test=True)
    elif args.net == 'mb1-ssd-lite-2197':
        net = create_mobilenetv1_ssd_lite_2197(len(class_names),  width_mult=1.0 ,is_test=True)
    elif args.net == 'mb1-ssd-lite-3184':
        net = create_mobilenetv1_ssd_lite_3184(len(class_names),  width_mult=1.0 ,is_test=True)
    elif args.net == 'mb1-ssd-lite-5693':
        net = create_mobilenetv1_ssd_lite_5693(len(class_names),  width_mult=1.0 ,is_test=True)
    elif args.net == 'mb1-ssd-lite-025extra-224':
        net = create_mobilenetv1_ssd_lite_025extra_224(len(class_names),  width_mult=args.width_mult ,is_test=True)
    elif args.net == 'mb1-ssd-lite-025extra-4box':
        net = create_mobilenetv1_ssd_lite_025extra_4box(len(class_names),  width_mult=args.width_mult ,is_test=True)
    elif args.net == 'mb1-ssd-lite-025extra-4box-same':
        net = create_mobilenetv1_ssd_lite_025extra_4box_same(len(class_names),  width_mult=args.width_mult ,is_test=True)
        
    elif args.net == 'mb1-ssd-lite-4box':
        net = create_mobilenetv1_ssd_lite_4box(len(class_names),  width_mult=args.width_mult ,is_test=True)
    elif args.net == 'mb1-ssd-lite-025extra-3box':
        net = create_mobilenetv1_ssd_lite_025extra_3box(len(class_names),  width_mult=args.width_mult ,is_test=True)
    elif args.net == 'mb1-ssd-lite-025extra-384':
        net = create_mobilenetv1_ssd_lite_025extra384(len(class_names),  width_mult=args.width_mult ,is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    elif args.net == 'mb2-ssd-lite-v2':
        net = create_mobilenetv2_ssd_lite_v2(len(class_names), width_mult=args.mb2_width_mult, is_test=True)        
    elif args.net == 'mb3-large-ssd-lite':
        net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb3-small-ssd-lite':
        net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')
    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite' or args.net == "mb1-ssd-lite-025extra" :
        predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite-025extra-224':
        predictor = create_mobilenetv1_ssd_lite_predictor224(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite-277' or 'mb1-ssd-lite-303':
        predictor = create_mobilenetv1_ssd_lite_predictor_277(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite-5693':
        predictor = create_mobilenetv1_ssd_lite_predictor_5693(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite-1679':
        predictor = create_mobilenetv1_ssd_lite_predictor_1679(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite-398':
        predictor = create_mobilenetv1_ssd_lite_predictor_398(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite-830':
        predictor = create_mobilenetv1_ssd_lite_predictor_830(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite-2197':
        predictor = create_mobilenetv1_ssd_lite_predictor_2197(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite-3184':
        predictor = create_mobilenetv1_ssd_lite_predictor_3184(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite-025extra-4box' or  args.net == 'mb1-ssd-lite-025extra-4box':
        predictor = create_mobilenetv1_ssd_lite_predictor4box(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite-025extra-3box':
        predictor = create_mobilenetv1_ssd_lite_predictor_3box(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite-025extra-384':
        predictor = create_mobilenetv1_ssd_lite_predictor384(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net,nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb2-ssd-lite' or args.net == "mb3-large-ssd-lite" or args.net == "mb3-small-ssd-lite" or 'mb2-ssd-lite-v2':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    results = []
    for i in range(len(dataset)):
        print("process image", i)
        timer.start("Load Image")
        image = dataset.get_image(i)
        print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            args.iou_threshold,
            args.use_2007_metric
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")

    print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
