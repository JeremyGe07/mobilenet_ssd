import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
# from cifar_resnet import ResNet18
# import cifar_resnet as resnet

import torch_pruning as tp
import argparse
import torch
# from torchvision.datasets import CIFAR10
# from torchvision import transforms
# import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 
from vision.ssd.mobilenetv1_ssd_lite_025extra import create_mobilenetv1_ssd_lite_025extra

from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.ssd import MatchPrior,SSD
from vision.datasets.voc_dataset import VOCDataset
from vision.utils.misc import  store_labels,str2bool, Timer
from torch.utils.data import DataLoader, ConcatDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.utils import box_utils, measurements
from vision.ssd.config import mobilenetv1_ssd_config
import pathlib
from torch.nn import Conv2d, Sequential, ModuleList, ReLU


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--round', type=int, default=1)
parser.add_argument("--label_file", type=str, default='models/voc-model-labels.txt', help="The label file path.")
parser.add_argument('--dataset', type=str, default='./VOC2007')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--t_max', default=120, type=float,)
parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--prune_folder', default='models/prune/',
                    help='Directory for saving prune models')
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--loss', default="smoothl1", type=str,
                    help="loss option: smoothl1 or diou ")
parser.add_argument("--trained_model", type=str)

args = parser.parse_args()



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

# def get_dataloader():
#     train_loader = torch.utils.data.DataLoader(
#         CIFAR10('./data', train=True, transform=transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ]), download=True),batch_size=args.batch_size, num_workers=2)
#     test_loader = torch.utils.data.DataLoader(
#         CIFAR10('./data', train=False, transform=transforms.Compose([
#             transforms.ToTensor(),
#         ]),download=True),batch_size=args.batch_size, num_workers=2)
#     return train_loader, test_loader

# def eval(model, test_loader):
    
#     true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(test_loader)

    
#     correct = 0
#     total = 0
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#     with torch.no_grad():
#         for i, (img, target) in enumerate(test_loader):
#             img = img.to(device)
#             out = model(img)
#             pred = out.max(1)[1].detach().cpu().numpy()
#             target = target.cpu().numpy()
#             correct += (pred==target).sum()
#             total += len(target)
#     return correct / total

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
    
        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
    
        optimizer.step()
    
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            print(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss
            # TODO: loss = 2*regression_loss + classification_loss and change the above loss in train()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

def train_model(model, train_loader, test_loader):
    config = mobilenetv1_ssd_config
    debug_steps=100
    last_epoch = -1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=device, loss=args.loss)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                        weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.t_max)
    model.to(device)

    for epoch in range(last_epoch + 1, args.total_epochs):
        scheduler.step()
        train(train_loader, model, criterion, optimizer,
              device=device, debug_steps=debug_steps, epoch=epoch)
        
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(test_loader, model, criterion, device)
            print(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
            model.save(model_path)
            print(f"Saved model {model_path}")
    # best_acc = -1
        # model.eval()
        # acc = eval(model, test_loader)
        # print("Epoch %d/%d, Acc=%.4f"%(epoch, args.total_epochs, acc))
        # if best_acc<acc:
        #     torch.save( model, 'resnet18-round%d.pth'%(args.round) )
        #     best_acc=acc
            # val_loss = test(test_loader, model, criterion, device)[0]
            # print("Epoch %d/%d, Validation Loss=%.4f"%(epoch, args.total_epochs, val_loss))
            # # if best_acc<acc:
            # torch.save( model, 'ssd-round%d.pth'%(args.round) )
            # best_acc=acc

        # scheduler.step()
    # print("Best Acc=%.4f"%(best_acc))
    
    
    

def prune_model(model):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 32, 32) )
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 300, 300) )

    def prune_conv(conv, amount=0.6):
        #weight = conv.weight.detach().cpu().numpy()
        #out_channels = weight.shape[0]
        #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        #num_pruned = int(out_channels * pruned_prob)
        #pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()
    
    block_prune_probs = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    blk_id = 0
    last_headers=['regression_headers.0.2','regression_headers.1.2','regression_headers.2.2','regression_headers.3.2',
                  'regression_headers.4.2','regression_headers.5','classification_headers.0.2','classification_headers.1.2',
                  'classification_headers.2.2','classification_headers.3.2','classification_headers.4.2','classification_headers.5']
    for k,m in model.named_modules():
        if isinstance( m, nn.Conv2d ) and k not in last_headers:
            prune_conv( m, 0.2 )
            # prune_conv( m.conv1, block_prune_probs[blk_id] )
            # prune_conv( m.conv2, block_prune_probs[blk_id] )
            blk_id+=1
    return model    

def main():
    # train_loader, test_loader = get_dataloader()
    config = mobilenetv1_ssd_config
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    train_dataset = VOCDataset(args.dataset, transform=train_transform,
                         target_transform=target_transform)
    label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
    store_labels(label_file, train_dataset.class_names)
    num_classes = config.num_classes
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)

    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    # class_names = [name.strip() for name in open(args.label_file).readlines()]
    class_names = ['BACKGROUND','person']

    val_dataset = VOCDataset(args.dataset, transform=test_transform,
                          target_transform=target_transform, is_test=True)
    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    test_dataset = VOCDataset(args.dataset,is_test=True)
    test_loader = test_dataset
    
    
    if args.mode=='train':
        args.round=0
        model = create_mobilenetv1_ssd_lite_025extra(2,  width_mult=0.15 ,is_test=False)

        train_model(model, train_loader, val_loader)
    elif args.mode=='prune':
        previous_ckpt = args.trained_model
        print("Pruning round %d, load model from %s"%( args.round, previous_ckpt ))
        model = create_mobilenetv1_ssd_lite_025extra(2,  width_mult=0.15 ,is_test=False)
        model.load(args.trained_model)
        prune_model(model)
        print(model)
        model_path = os.path.join(args.prune_folder, f"prune.pth")
        model.save(model_path)
        print(f"Saved prune model {model_path}")
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        train_model(model, train_loader, test_loader)
    # elif args.mode=='test':
    #     ckpt = 'resnet18-round%d.pth'%(args.round)
    #     print("Load model from %s"%( ckpt ))
    #     model = torch.load( ckpt )
    #     params = sum([np.prod(p.size()) for p in model.parameters()])
    #     print("Number of Parameters: %.1fM"%(params/1e6))
    #     acc = eval(model, test_loader)
    #     print("Acc=%.4f\n"%(acc))

if __name__=='__main__':
    main()
