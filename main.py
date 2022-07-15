from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

# from vgg import vgg
import shutil

from vision.ssd.mobilenetv1_ssd_lite_025extra import create_mobilenetv1_ssd_lite_025extra
from vision.ssd.config import mobilenetv1_ssd_config
from torch.nn import Conv2d, Sequential, ModuleList, ReLU
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.ssd import MatchPrior
from vision.datasets.voc_dataset import VOCDataset
from vision.utils.misc import  store_labels
from torch.utils.data import DataLoader, ConcatDataset
from vision.nn.multibox_loss import MultiboxLoss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--datasets', nargs='+', default=['./VOC2007'],help='Dataset directory path')

parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='refine from prune model')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--pretrain_model', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_dataset', default='./VOC2007',help='Dataset directory path')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # !may not be same as .cuda()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

config = mobilenetv1_ssd_config

train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
target_transform = MatchPrior(config.priors, config.center_variance,
                              config.size_variance, 0.5)

test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

datasets = []
for dataset_path in args.datasets:
    dataset = VOCDataset(dataset_path, transform=train_transform,
                         target_transform=target_transform)
    label_file = os.path.join("models/voc-model-labels.txt")
    # label_file = os.path.join("D:/GJZ/Documents/shixi/pedestrian detect/oldfinal/models/voc-model-labels.txt")

    store_labels(label_file, dataset.class_names)
    # num_classes = len(dataset.class_names)
    num_classes = config.num_classes

    datasets.append(dataset)
train_dataset = ConcatDataset(datasets)
train_loader = DataLoader(train_dataset, args.batch_size,
                          num_workers=args.num_workers,
                          shuffle=True)
val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                         target_transform=target_transform, is_test=True)
test_loader = DataLoader(val_dataset, args.batch_size,
                        num_workers=1,
                        shuffle=False)

if args.refine:
    checkpoint = torch.load(args.refine)
    model = create_mobilenetv1_ssd_lite_025extra(len(['Background','person']),  width_mult=0.25 ,is_test=True)

    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
else:
    model = create_mobilenetv1_ssd_lite_025extra(len(['Background','person']),  width_mult=0.25 ,is_test=True)
    if args.pretrain_model:
        model.load(args.pretrain_model)
    elif args.base_net:
        model.init_from_base_net(args.base_net,args.width_mult)
    elif args.pretrained_ssd:
        model.init_from_pretrained_ssd(args.pretrained_ssd)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):  # can be relu? read yolo prune folder
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1


def train(epoch):
    model.train()
    criterion = MultiboxLoss(mobilenetv1_ssd_config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE, loss='smoothl1')
    for batch_idx, (images, boxes, labels) in enumerate(train_loader):
        # images = images.to(device)
        # boxes = boxes.to(device)
        # labels = labels.to(device)
        if args.cuda:
            images, boxes, labels = images.cuda(), boxes.cuda(), labels.cuda()
        # data, target = Variable(data), Variable(target)
        images, boxes, labels = Variable(images), Variable(boxes), Variable(labels)
        optimizer.zero_grad()
        confidence, locations = model(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    criterion = MultiboxLoss(mobilenetv1_ssd_config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE, loss='smoothl1')
    for _, data in enumerate(test_loader):
        images, boxes, labels = data
        if args.cuda:
            images, boxes, labels = images.cuda(), boxes.cuda(), labels.cuda()
        images, boxes, labels = Variable(images, volatile=True), Variable(boxes), Variable(labels)
        confidence, locations = model(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        # test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        num += 1

    # test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(running_loss / num))
    return running_loss / num
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    # return correct / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    prec1 = test()
    # is_best = prec1 > best_prec1
    # best_prec1 = max(prec1, best_prec1)
    is_best = prec1 < best_prec1
    best_prec1 = min(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best)
    
# def test(loader, net, criterion, device):
#     net.eval()
#     running_loss = 0.0
#     running_regression_loss = 0.0
#     running_classification_loss = 0.0
#     num = 0
#     for _, data in enumerate(loader):
#         images, boxes, labels = data
#         images = images.to(device)
#         boxes = boxes.to(device)
#         labels = labels.to(device)
#         num += 1

#         with torch.no_grad():
#             confidence, locations = net(images)
#             regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
#             loss = regression_loss + classification_loss
#             # TODO: loss = 2*regression_loss + classification_loss and change the above loss in train()

#         running_loss += loss.item()
#         running_regression_loss += regression_loss.item()
#         running_classification_loss += classification_loss.item()
#     return running_loss / num, running_regression_loss / num, running_classification_loss / num
