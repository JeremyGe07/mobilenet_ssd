import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F
from collections import OrderedDict 

from ..utils import box_utils
from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #


class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model, width):
        # if width == 1:
        #     self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        # else:
        base_weights = torch.load(model, map_location=lambda storage, loc: storage) 
        new_state_dict = OrderedDict()
        for k, v in base_weights.items():
            #for mbv2
            # name = k
            # name = k[9:]   # remove `model.`，即只取model.0.weights的后面几位
            # if k == 'conv.0.weight':
            #     name = '18.0.weight'
            # elif k == 'conv.1.weight':
            #     name = '18.1.weight'
            # elif k == 'conv.1.bias':
            #     name = '18.1.bias'
            # elif k == 'conv.1.running_mean':
            #     name = '18.1.running_mean'
            # elif k == 'conv.1.running_var':
            #     name = '18.1.running_var'
            # elif k == 'conv.1.num_batches_tracked':
            #     name = '18.1.num_batches_tracked'
            # elif k == 'classifier.weight':
            #     continue
            # elif k == 'classifier.bias':
            #     continue
        #     new_state_dict[name] = v 
        # self.base_net.load_state_dict(new_state_dict)
            
            # for mbv1
            name = k[6:]   # remove `model.`，即只取model.0.weights的后面几位

        if name == '0.2.bias':
            name = '0.1.bias'
        elif name == '0.2.running_mean':
            name = '0.1.running_mean'
        elif name == '0.2.running_var':
            name = '0.1.running_var'
        elif name == '0.1.weight':
            name = '0.0.weight'
        elif name == '0.2.weight':
            name = '0.1.weight'
            
        elif name == '2.1.weight':
            name = '2.0.weight'
        elif name == '2.2.weight':
            name = '2.1.weight'
        elif name == '2.2.bias':
            name = '2.1.bias'
        elif name == '2.2.running_mean':
            name = '2.1.running_mean'
        elif name == '2.2.running_var':
            name = '2.1.running_var'
        elif name == '2.5.weight':
            name = '2.4.weight'
        elif name == '2.5.bias':
            name = '2.4.bias'
        elif name == '2.5.running_mean':
            name = '2.4.running_mean'
        elif name == '2.5.running_var':
            name = '2.4.running_var'
        elif name == '2.4.weight':
            name = '2.3.weight'
        
        elif name == '4.1.weight':
            name = '4.0.weight'
        elif name == '4.2.weight':
            name = '4.1.weight'
        elif name == '4.2.bias':
            name = '4.1.bias'
        elif name == '4.2.running_mean':
            name = '4.1.running_mean'
        elif name == '4.2.running_var':
            name = '4.1.running_var'
        elif name == '4.5.weight':
            name = '4.4.weight'
        elif name == '4.5.bias':
            name = '4.4.bias'
        elif name == '4.5.running_mean':
            name = '4.4.running_mean'
        elif name == '4.5.running_var':
            name = '4.4.running_var'
        elif name == '4.4.weight':
            name = '4.3.weight'
        
        elif name == '6.1.weight':
            name = '6.0.weight'
        elif name == '6.2.weight':
            name = '6.1.weight'
        elif name == '6.2.bias':
            name = '6.1.bias'
        elif name == '6.2.running_mean':
            name = '6.1.running_mean'
        elif name == '6.2.running_var':
            name = '6.1.running_var'
        elif name == '6.5.weight':
            name = '6.4.weight'
        elif name == '6.5.bias':
            name = '6.4.bias'
        elif name == '6.5.running_mean':
            name = '6.4.running_mean'
        elif name == '6.5.running_var':
            name = '6.4.running_var'
        elif name == '6.4.weight':
            name = '6.3.weight'
            
        elif name == '12.1.weight':
            name = '12.0.weight'
        elif name == '12.2.weight':
            name = '12.1.weight'
        elif name == '12.2.bias':
            name = '12.1.bias'
        elif name == '12.2.running_mean':
            name = '12.1.running_mean'
        elif name == '12.2.running_var':
            name = '12.1.running_var'
        elif name == '12.5.weight':
            name = '12.4.weight'
        elif name == '12.5.bias':
            name = '12.4.bias'
        elif name == '12.5.running_mean':
            name = '12.4.running_mean'
        elif name == '12.5.running_var':
            name = '12.4.running_var'
        elif name == '12.4.weight':
            name = '12.3.weight'
            
            new_state_dict[name] = v 
            self.base_net.load_state_dict(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_kaiming_init_)
        self.regression_headers.apply(_kaiming_init_)

    def init(self):
        self.base_net.apply(_kaiming_init_)
        self.source_layer_add_ons.apply(_kaiming_init_)
        self.extras.apply(_kaiming_init_)
        self.classification_headers.apply(_kaiming_init_)
        self.regression_headers.apply(_kaiming_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

def _kaiming_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.xavier_uniform_(m.weight)
