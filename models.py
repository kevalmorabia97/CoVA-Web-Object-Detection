import numpy as np
import torch
import torch.nn as nn
import torchvision

from utils import count_parameters


class WebObjExtractionNet(nn.Module):
    def __init__(self, roi_output_size, img_H, n_classes, backbone='resnet', trainable_convnet=False, class_names=None):
        """
        Args:
            roi_output_size: Tuple (int, int) which will be output of the roi_pool layer for each channel of convnet_feature
            img_H: height of image given as input to the convnet. Image assumed to be of same W and H
            n_classes: num of classes for BBoxes
            backbone: string stating which convnet feature extractor to use.
                      Allowed backbones are [alexnet, resnet (default)]
            trainable_convnet: if True then convnet weights will be modified while training (default: False)
            class_names: list of n_classes string elements containing names of the classes (default: [0, 1, ..., n_classes-1])
        """
        print('Initializing WebObjExtractionNet model...')
        super(WebObjExtractionNet, self).__init__()
        self.n_classes = n_classes
        self.backbone = backbone
        self.trainable_convnet = trainable_convnet
        if class_names is None:
            self.class_names = np.arange(self.n_classes).astype(str)
        else:
            self.class_names = class_names
        
        if self.backbone not in ['alexnet', 'resnet']:
            self.backbone = 'resnet'
            print('Invalid backbone provided. Setting backbone to Resnet')
            
        if self.backbone == 'resnet':
            self.convnet = torchvision.models.resnet18(pretrained=True)
            modules = list(self.convnet.children())[:-5] # remove last few layers!
        elif self.backbone == 'alexnet':
            self.convnet = torchvision.models.alexnet(pretrained=True)
            modules = list(self.convnet.features.children())[:7] # remove last few layers!
        
        print('Using first few layers of \"%s\" as ConvNet Visual Feature Extractor' % self.backbone)
        
        self.convnet = nn.Sequential(*modules)
        if self.trainable_convnet == False:
            print('ConvNet weights Freezed')
            for p in self.convnet.parameters(): # freeze weights
                p.requires_grad = False
        
        _imgs = torch.autograd.Variable(torch.Tensor(1, 3, img_H, img_H))
        _conv_feat = self.convnet(_imgs)
        _convnet_output_size = _conv_feat.size() # [1, C, H, W]
        n_feat = _convnet_output_size[1] * roi_output_size[0] * roi_output_size[1]
        
        spatial_scale = _convnet_output_size[2]/img_H
        self.roi_pool = torchvision.ops.RoIPool(roi_output_size, spatial_scale)
        self.fc = nn.Linear(n_feat, n_classes)
        
        print('ConvNet Feature Map size:', _convnet_output_size)
        print('Trainable parameters:', count_parameters(self))
        print(self)
        print('-'*50)
    
    def forward(self, images, bboxes):
        """
        Args:
            images: torch.Tensor of size [batch_size, 3, img_H, img_H]
            bboxes: list of torch.Tensor, each of size [n_bboxes, 4]
                each of the 4 values correspond to [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        
        Returns:
            prediction_scores: torch.Tensor of size [n_bboxes, n_classes]
        """
        conv_feat = self.convnet(images)
        pooled_feat = self.roi_pool(conv_feat, bboxes)
        pooled_feat = pooled_feat.view(pooled_feat.size()[0],-1)
        output = self.fc(pooled_feat)

        return output