import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils import count_parameters


class WebObjExtractionNet(nn.Module):
    def __init__(self, roi_output_size, img_H, n_classes, backbone='alexnet', use_attention=True, hidden_dim=300,
                 trainable_convnet=True, drop_prob=0.2, use_bbox_feat=True, class_names=None):
        """
        Args:
            roi_output_size: Tuple (int, int) which will be output of the roi_pool layer for each channel of convnet_feature
            img_H: height of image given as input to the convnet. Image assumed to be of same W and H
            n_classes: num of classes for BBoxes
            backbone: string stating which convnet feature extractor to use. Allowed values: [alexnet (default), resnet]
            use_attention: if True, learn scores for all 2*context_size contexts and take weighted avg for context_representation 
            hidden_dim: size of hidden contextual representation, used when use_attention=True (default: 300)
            trainable_convnet: if True then convnet weights will be modified while training (default: True)
            drop_prob: dropout probability (default: 0.2)
            use_bbox_feat: if True, then concatenate x,y,w,h with convnet visual features for classification of a BBox (default: True)
            class_names: list of n_classes string elements containing names of the classes (default: [0, 1, ..., n_classes-1])
        """
        print('Initializing WebObjExtractionNet...')
        super(WebObjExtractionNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.use_attention = use_attention
        self.trainable_convnet = trainable_convnet
        self.use_bbox_feat = use_bbox_feat
        self.class_names = np.arange(self.n_classes).astype(str) if class_names is None else class_names

        if backbone == 'resnet':
            self.convnet = torchvision.models.resnet18(pretrained=True)
            modules = list(self.convnet.children())[:-5] # remove last few layers!
        elif backbone == 'alexnet':
            self.convnet = torchvision.models.alexnet(pretrained=True)
            modules = list(self.convnet.features.children())[:7] # remove last few layers!

        self.convnet = nn.Sequential(*modules)
        if self.trainable_convnet == False:
            for p in self.convnet.parameters(): # freeze weights
                p.requires_grad = False

        _imgs = torch.autograd.Variable(torch.Tensor(1, 3, img_H, img_H))
        _conv_feat = self.convnet(_imgs)
        _convnet_output_size = _conv_feat.shape # [1, C, H, W]
        spatial_scale = _convnet_output_size[2]/img_H

        self.roi_pool = torchvision.ops.RoIPool(roi_output_size, spatial_scale)

        self.n_visual_feat = _convnet_output_size[1] * roi_output_size[0] * roi_output_size[1]
        self.n_context_feat = self.hidden_dim if use_attention else self.n_visual_feat
        self.n_bbox_feat = 4 if self.use_bbox_feat else 0 # x,y,w,h of BBox
        self.n_total_feat = self.n_visual_feat + self.n_context_feat + self.n_bbox_feat

        if self.use_attention:
            self.encoder = nn.Sequential(
                nn.Linear(self.n_visual_feat, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

            self.attention_layer = nn.Linear(2*self.hidden_dim, 1)
            with torch.no_grad():
                print('Attention layer weights initialized to 0')
                self.attention_layer.weight.fill_(0)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.n_total_feat, self.n_total_feat),
            nn.BatchNorm1d(self.n_total_feat),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.n_total_feat, self.n_classes),
        )

        print(self)
    
    def forward(self, images, bboxes, context_indices):
        """
        Args:
            images: torch.Tensor of size [batch_size, 3, img_H, img_H]
            bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
                each each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            context_indices: [total_n_bboxes_in_batch, 2 * context_size] i.e. bbox indices (0-indexed) of contexts for all n bboxes.
                If not enough found, rest are -1
        
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        batch_size = bboxes.shape[0]
        context_size = int(context_indices.shape[1]/2)

        ##### OWN VISUAL FEATURES #####
        conv_feat = self.convnet(images)
        own_features = self.roi_pool(conv_feat, bboxes).view(batch_size,-1)

        ##### CONTEXT VISUAL FEATURES USING ATTENTION #####
        zero_feat = torch.zeros(self.n_visual_feat).view(1,-1).to(images.device) # for -1 contexts i.e. extra padded
        pooled_feat_padded = torch.cat((own_features, zero_feat), dim=0)
        context_features = pooled_feat_padded[context_indices.view(-1)].view(batch_size,-1) # [batch_size, 2 * context_size * n_visual_feat]

        if self.use_attention:
            own_features_encoded = self.encoder(own_features)

            context_outputs = []
            attention_wts = []
            for i in range(2*context_size):
                curr_features_encoded = self.encoder(context_features[:, i*self.n_visual_feat:(i+1)*self.n_visual_feat])
                concatenated_feat = torch.cat((own_features_encoded, curr_features_encoded), dim=1)
                curr_attention_wt = self.attention_layer(concatenated_feat) # [batch_size, 1]
                
                context_outputs.append(curr_features_encoded)
                attention_wts.append(curr_attention_wt)

            context_outputs = torch.cat(context_outputs, dim=1).view(batch_size, 2*context_size, -1) # [batch_size, 2 * context_size, hidden_dim]
            attention_wts = torch.softmax(torch.cat(attention_wts, dim=1), dim=1).unsqueeze(-1) # [batch_size, 2 * context_size, 1]
            
            context_representation = (context_outputs * attention_wts).sum(1) # [batch_size, hidden_dim]
        else: # average of context features for context representation
            context_representation = context_features.view(batch_size, -1, self.n_visual_feat).sum(1) # [batch_size, n_visual_feat]
            context_representation = context_representation/(context_indices != -1).sum(1).view(batch_size,-1)

        ##### BBOX FEATURES #####
        if self.use_bbox_feat:
            bbox_features = bboxes[:, 1:].clone()
            bbox_features[:, 2:] -= bbox_features[:, :2] # convert to [top_left_x, top_left_y, width, height]
        else:
            bbox_features = bboxes[:, :0] # size [n_bboxes, 0]

        ##### FINAL FEATURE VECTOR #####
        combined_feat = torch.cat((own_features, context_representation, bbox_features), dim=1)
        output = self.decoder(combined_feat)

        return output