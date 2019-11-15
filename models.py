import numpy as np
import torch
import torch.nn as nn
import torchvision

from utils import count_parameters


class WebObjExtractionNet(nn.Module):
    def __init__(self, roi_output_size, img_H, n_classes, backbone='alexnet', trainable_convnet=False, use_pos_feat=False, class_names=None):
        """
        Args:
            roi_output_size: Tuple (int, int) which will be output of the roi_pool layer for each channel of convnet_feature
            img_H: height of image given as input to the convnet. Image assumed to be of same W and H
            n_classes: num of classes for BBoxes
            backbone: string stating which convnet feature extractor to use. Allowed values: [alexnet (default), resnet]
            trainable_convnet: if True then convnet weights will be modified while training (default: False)
            use_pos_feat: if True, then concatenate x,y,w,h (normalized from 0 to 1) with convnet visual features for classification of a BBox (default: False)
            class_names: list of n_classes string elements containing names of the classes (default: [0, 1, ..., n_classes-1])
        """
        print('Initializing WebObjExtractionNet...')
        super(WebObjExtractionNet, self).__init__()

        self.img_H = img_H
        self.n_classes = n_classes
        self.backbone = backbone
        self.trainable_convnet = trainable_convnet
        self.use_pos_feat = use_pos_feat
        self.class_names = np.arange(self.n_classes).astype(str) if class_names is None else class_names

        if self.backbone not in ['alexnet', 'resnet']:
            self.backbone = 'alexnet'
            print('---> Invalid backbone provided. Setting backbone to Alexnet')
            
        if self.backbone == 'resnet':
            self.convnet = torchvision.models.resnet18(pretrained=True)
            modules = list(self.convnet.children())[:-5] # remove last few layers!
        elif self.backbone == 'alexnet':
            self.convnet = torchvision.models.alexnet(pretrained=True)
            modules = list(self.convnet.features.children())[:7] # remove last few layers!

        print('Using first few layers of \"%s\" as Visual Feature Extractor' % self.backbone)        
        self.convnet = nn.Sequential(*modules)
        if self.trainable_convnet == False:
            print('---> ConvNet weights Freezed')
            for p in self.convnet.parameters(): # freeze weights
                p.requires_grad = False

        _imgs = torch.autograd.Variable(torch.Tensor(1, 3, self.img_H, self.img_H))
        _conv_feat = self.convnet(_imgs)
        _convnet_output_size = _conv_feat.size() # [1, C, H, W]
        spatial_scale = _convnet_output_size[2]/self.img_H

        self.n_visual_feat = _convnet_output_size[1] * roi_output_size[0] * roi_output_size[1]
        self.n_pos_feat = 4 if self.use_pos_feat else 0 # x,y,w,h of BBox
        self.n_feat = self.n_visual_feat + self.n_pos_feat
        
        self.roi_pool = torchvision.ops.RoIPool(roi_output_size, spatial_scale)
        self.bn = nn.BatchNorm1d(self.n_feat)
        self.fc = nn.Linear(self.n_feat, n_classes)
        
        print('ConvNet Feature Map size:', _convnet_output_size)
        print('Trainable parameters:', count_parameters(self))
        print('-'*50)
        print(self)
        print('-'*50)
    
    def forward(self, images, bboxes):
        """
        Args:
            images: torch.Tensor of size [batch_size, 3, img_H, img_H]
            bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
                each each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        ##### VISUAL FEATURES #####
        conv_feat = self.convnet(images)
        pooled_feat = self.roi_pool(conv_feat, bboxes)
        pooled_feat = pooled_feat.view(pooled_feat.size()[0],-1)

        ##### POSITION FEATURES #####
        pos_feat = bboxes[:, :0] # size [n_bboxes, 0]
        if self.use_pos_feat:
            pos_feat = bboxes[:, 1:].clone()
            pos_feat[:, 2:] -= pos_feat[:, :2] # convert to [top_left_x, top_left_y, width, height]

        ##### FINAL FEATURE VECTOR #####
        combined_feat = torch.cat((pooled_feat, pos_feat), dim=1)
        combined_feat = self.bn(combined_feat)
        output = self.fc(combined_feat)

        return output