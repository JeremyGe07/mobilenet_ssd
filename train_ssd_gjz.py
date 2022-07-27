import argparse
import os
import logging
import sys
import itertools

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR,CosineAnnealingWarmRestarts
from timm.scheduler import CosineLRScheduler

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenetv1_ssd_lite_025extra import create_mobilenetv1_ssd_lite_025extra
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config, mbv1_ssd_224_config, mbv1_ssd_config_4box,mobilenetv1_ssd_384_config,mobilenetv1_ssd_config_3box,mbv1_ssd_config_4box_same
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.mobilenetv1_ssd_lite_224 import create_mobilenetv1_ssd_lite_025extra_224
from vision.ssd.mobilenetv1_ssd_lite_025extra_4box import create_mobilenetv1_ssd_lite_025extra_4box
from vision.ssd.mobilenetv1_ssd_lite_025extra_384 import create_mobilenetv1_ssd_lite_025extra384
from vision.ssd.mobilenetv1_ssd_lite_025extra_3box import create_mobilenetv1_ssd_lite_025extra_3box
from vision.ssd.mobilenetv1_ssd_lite_277kb import create_mobilenetv1_ssd_lite_277, create_mobilenetv1_ssd_lite_predictor_277
from vision.ssd.mobilenet_v2_ssd_lite_v2 import create_mobilenetv2_ssd_lite_v2
from vision.ssd.mobilenetv1_ssd_lite_025extra_4box_same import create_mobilenetv1_ssd_lite_025extra_4box_same
from vision.ssd.mobilenetv1_ssd_lite_4box import create_mobilenetv1_ssd_lite_4box


#TODO try224input --net mb1-ssd-lite-025extra-224; try 4box --net mb1-ssd-lite-025extra-4box; try  change box ratio from [2,3] to [1.7,2.6] base on 4box --net mb1-ssd-lite-025extra-4box
# nohup python train_ssd.py --dataset_type voc --datasets VOC2007my --net mb1-ssd-lite-025extra --base_net mobilenet_v1_size_224_alpha_0.25_top.pth --lr 0.01 --batch_size 8 --num_epochs 400 --scheduler cosine --t_max 400 --num_workers 4 --checkpoint_folder 'models/ws0.25tmax400extra0.25' --width_mult 0.25 1>models/ws0.25tmax400extra0.25/logs.txt 2>models/ws0.25tmax400extra0.25/warns.txt
# nohup python train_ssd.py --dataset_type voc --datasets VOC2007 --net mb1-ssd-lite-025extra --base_net mobilenet_v1_size_224_alpha_0.25_top.pth --lr 0.01 --batch_size 8 --num_epochs 400 --scheduler cosine --t_max 400 --num_workers 4 --width_mult 0.25 --checkpoint_folder 'models/' --gpu '1' -sr  1>models//logs.txt 2>models//warns.txt
#new: nohup python train_ssd_gjz.py --dataset_type voc --datasets VOC2007 --net mb1-ssd-lite-025extra --base_net mobilenet_v1_size_224_alpha_0.25_top.pth --lr 0.01 --batch_size 8 --num_epochs 400 --scheduler cosine --t_max 400 --num_workers 4 --checkpoint_folder 'DDDDD' --width_mult 0.25  --validation_dataset VOC2007
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')

parser.add_argument('--datasets', nargs='+', default=['./VOCdevkit'],help='Dataset directory path')
parser.add_argument('--validation_dataset', default='./VOC2007',help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")


parser.add_argument('--net', default="mb1-ssd-lite",
                    help="The network architecture, it can be mb1-ssd, ,mb1-ssd-lite-025extra, mb2-ssd-lite, mb3-large-ssd-lite, mb3-small-ssd-lite or vgg16-ssd.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

parser.add_argument('--width_mult', default=1.0, type=float,
                    help='Batch size for training')


# Params for SGD
parser.add_argument('--optimizer', default="SGD", type=str,help='AdamW,SGD')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")
# Params for Location Loss computation
parser.add_argument('--loss', default="smoothl1", type=str,
                    help="loss option: smoothl1 or diou ")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')

parser.add_argument('--gpu', default='0', type=str,
                    help='use which gpu to train, if >2, use comma to separate')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--last_epoch', type=float, default=-1,
                    help='last_epoch')
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()





#多块使用逗号隔开
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")



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
        
        if args.sr:
            updateBN()
            
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
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

def updateBN():
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):  # can be relu? read yolo prune folder
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

if __name__ == '__main__':
    timer = Timer()

    logging.info(args)
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        #create_net = create_mobilenetv1_ssd_lite
        create_net = lambda num: create_mobilenetv1_ssd_lite(num, width_mult=args.width_mult)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite-025extra':
        #create_net = create_mobilenetv1_ssd_lite
        create_net = lambda num: create_mobilenetv1_ssd_lite_025extra(num, width_mult=args.width_mult)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite-025extra-224':
        #create_net = create_mobilenetv1_ssd_lite
        create_net = lambda num: create_mobilenetv1_ssd_lite_025extra_224(num, width_mult=args.width_mult)
        config = mbv1_ssd_224_config    
    elif args.net == 'mb1-ssd-lite-025extra-4box':
        #create_net = create_mobilenetv1_ssd_lite
        create_net = lambda num: create_mobilenetv1_ssd_lite_025extra_4box(num, width_mult=args.width_mult)
        config = mbv1_ssd_config_4box
    elif args.net == 'mb1-ssd-lite-025extra-4box-same':
        #create_net = create_mobilenetv1_ssd_lite
        create_net = lambda num: create_mobilenetv1_ssd_lite_025extra_4box_same(num, width_mult=args.width_mult)
        config = mbv1_ssd_config_4box_same
    elif args.net == 'mb1-ssd-lite-025extra-384':
         #create_net = create_mobilenetv1_ssd_lite
         create_net = lambda num: create_mobilenetv1_ssd_lite_025extra384(num, width_mult=args.width_mult)
         config = mobilenetv1_ssd_384_config
    elif args.net == 'mb1-ssd-lite-025extra-3box':
         #create_net = create_mobilenetv1_ssd_lite
         create_net = lambda num: create_mobilenetv1_ssd_lite_025extra_3box(num, width_mult=args.width_mult)
         config = mobilenetv1_ssd_config_3box
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb2-ssd-lite-v2':
        create_net = lambda num: create_mobilenetv2_ssd_lite_v2(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3-large-ssd-lite':
        create_net = lambda num: create_mobilenetv3_large_ssd_lite(num)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3-small-ssd-lite':
        create_net = lambda num: create_mobilenetv3_small_ssd_lite(num)
        config = mobilenetv1_ssd_config
    # For testing whether prune is actually working
    elif args.net == 'mb1-ssd-lite-277':
        create_net = lambda num: create_mobilenetv1_ssd_lite_277(num, width_mult=1.0)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite-4box':
        create_net = lambda num: create_mobilenetv1_ssd_lite_4box(num, width_mult=1.0)
        config = mbv1_ssd_config_4box_same
        
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            # num_classes = len(dataset.class_names)
            num_classes = config.num_classes
        elif args.dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path,
                 transform=train_transform, target_transform=target_transform,
                 dataset_type="train", balance_data=args.balance_data)
            label_file = os.path.join(args.checkpoint_folder, "open-images-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            logging.info(dataset)
            # num_classes = len(dataset.class_names)
            num_classes = config.num_classes


        else:
            raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    logging.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif args.dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(dataset_path,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test")
        logging.info(val_dataset)
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    logging.info("Build network.")
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = args.last_epoch

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        if args.net == 'mb2-ssd-lite':
            net.init_from_base_net(args.base_net,args.mb2_width_mult)
        else:
            net.init_from_base_net(args.base_net,args.width_mult)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE, loss=args.loss)
    #optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
    #                            weight_decay=args.weight_decay)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        print('use AdamW ...')
        optimizer = torch.optim.AdamW(params, 
                                lr=args.lr, 
                                weight_decay=5e-2)     
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    elif args.scheduler == 'warm cosine1':
        logging.info("Uses CosineAnnealingWarmRestarts scheduler.")
        scheduler = CosineAnnealingWarmRestarts(optimizer, 5,79, last_epoch=last_epoch)    
    elif args.scheduler == 'warm cosine2':
        logging.info("Uses CosineAnnealingWarmRestarts scheduler.")
        scheduler = CosineLRScheduler(optimizer=optimizer, t_initial=200, lr_min=1e-5,warmup_t=4, warmup_lr_init=1e-4)  
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        if args.scheduler == 'warm cosine2':
            scheduler.step(epoch)
        else:
            scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")
