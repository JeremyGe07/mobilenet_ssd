import torch.nn as nn
import torch.nn.functional as F
import torch


from ..utils import box_utils

'''
假设一个训练样本中有2个 ground truth box，所有的 feature map 中获取的 prior box 一共有8732个。
那么可能分别有10、20个 prior box 能分别与这2个 ground truth box 匹配上，匹配成功的将会作为正样本参与分类和回归训练，
而未能匹配的则只会参加分类（负样本）训练。训练的损失包含分类损失和回归损失两部分。

将 prior box 和 grount truth box 按照IOU（JaccardOverlap）进行匹配，匹配成功则这个 prior box 就是 positive example（正样本），
如果匹配不上，就是 negative example（负样本），显然这样产生的负样本的数量要远远多于正样本。
这里默认做了难例挖掘：简单描述起来就是，将所有的匹配不上的 negative prior box 按照分类 loss 进行排序，
选择最高的 num_sel 个 prior box 序号集合作为最终的负样本集。这里就可以利用 num_sel 来控制最后正、负样本的比例在 1：3 左右。
'''

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce

#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss

class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device,loss):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)
        self.loss=loss

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.   每个不同尺寸的特征图上都会有groundtruth box吗,根据当前特征图尺度从原始图像中换算得到
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0] #这一步出了3000个框而不是2000，或者说loss的30000是因为前面的*6
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        # print('ooooooooooomask',mask.shape)
        # print('aaaaactual label',labels)
        # # print('mmmmmmmmmmask',mask)
        # print('connnnnnnfidden',confidence.reshape(-1, num_classes).shape)
        # print('llllllllllllabels',labels[mask].shape)
        # 分类损失包括：n个正样本损失，3n个负样本损失
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        # floss=FocalLoss()
        # classification_loss = floss(confidence.reshape(-1, num_classes), labels[mask])

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        #TODO 尝试使用 GIoU、DIoU、CIoU Loss 替换 Smooth L1 Loss。
        if self.loss == 'smoothl1':
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        elif self.loss == 'diou':
            smooth_l1_loss = diou(predicted_locations, gt_locations)

        num_pos = gt_locations.size(0)
        return smooth_l1_loss/num_pos, classification_loss/num_pos
        # return smooth_l1_loss/num_pos, classification_loss
# 
# DIoU Loss
def diou(bboxes1, bboxes2):
    # this is from official website:
    # https://github.com/Zzh-tju/CIoU/blob/master/layers/modules/multibox_loss.py
    bboxes1 = torch.sigmoid(bboxes1)        # make sure the input belongs to [0, 1]
    bboxes2 = torch.sigmoid(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True
    w1 = torch.exp(bboxes1[:, 2])       # this means this bbox has been encoded by log
    h1 = torch.exp(bboxes1[:, 3])       # you needn't do this if your bboxes are not encoded
    w2 = torch.exp(bboxes2[:, 2])
    h2 = torch.exp(bboxes2[:, 3])
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = bboxes1[:, 0]
    center_y1 = bboxes1[:, 1]
    center_x2 = bboxes2[:, 0]
    center_y2 = bboxes2[:, 1]

    inter_l = torch.max(center_x1 - w1 / 2, center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2, center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2, center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2, center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0)

    c_l = torch.min(center_x1 - w1 / 2, center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2, center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2, center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2, center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    c_diag = torch.clamp((c_r - c_l), min=0)**2 + torch.clamp((c_b - c_t), min=0)**2

    union = area1+area2-inter_area
    u = (inter_diag) / c_diag
    iou = inter_area / union
    dious = iou - u
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return torch.sum(1 - dious)
