import numpy as np
import torch
import torch.nn as nn
import torchvision

from utils import count_parameters


class WebObjExtractionNet(nn.Module):
    def __init__(self, roi_output_size, img_H, n_classes, backbone='alexnet', use_context=True, use_attention=True, n_attn_heads=8,
                 hidden_dim=384, use_bbox_feat=True, bbox_hidden_dim=32, trainable_convnet=True, drop_prob=0.2, class_names=None):
        """
        Args:
            roi_output_size: Tuple (int, int) which will be output of the roi_pool layer for each channel of convnet_feature
            img_H: height of image given as input to the convnet. Image assumed to be of same W and H
            n_classes: num of classes for BBoxes
            backbone: string stating which convnet feature extractor to use. Allowed values: [alexnet (default), resnet]
            use_context: if True, use context for context_representation along with own_features (default: True) 
            use_attention: if True, learn scores for all n_context contexts and take weighted avg for context_representation
                NOTE: this parameter is not used if use_context = False
            n_attn_heads: number of heads for self-attention, used when use_attention=True (default: 8)
            hidden_dim: size of hidden contextual representation, used when use_attention=True (default: 384)
            use_bbox_feat: if True, then use [x, y, w, h, asp_ratio] with convnet visual features for classification of a BBox (default: True)
            bbox_hidden_dim: size of hidden representation of 5 bbox features, used when use_bbox_feat=True (default: 32)
            trainable_convnet: if True then convnet weights will be modified while training (default: True)
            drop_prob: dropout probability (default: 0.2)
            class_names: list of n_classes string elements containing names of the classes (default: [0, 1, ..., n_classes-1])
        """
        super(WebObjExtractionNet, self).__init__()

        self.n_classes = n_classes
        self.use_context = use_context
        self.use_attention = use_attention
        self.n_attn_heads = n_attn_heads
        self.hidden_dim = hidden_dim
        self.use_bbox_feat = use_bbox_feat
        self.bbox_hidden_dim = bbox_hidden_dim
        self.class_names = np.arange(self.n_classes).astype(str) if class_names is None else class_names

        if backbone == 'resnet':
            self.convnet = torchvision.models.resnet18(pretrained=True)
            modules = list(self.convnet.children())[:-5] # remove last few layers!
        elif backbone == 'alexnet':
            self.convnet = torchvision.models.alexnet(pretrained=True)
            modules = list(self.convnet.features.children())[:7] # remove last few layers!

        self.convnet = nn.Sequential(*modules)
        if trainable_convnet == False:
            for p in self.convnet.parameters(): # freeze weights
                p.requires_grad = False

        _imgs = torch.autograd.Variable(torch.Tensor(1, 3, img_H, img_H))
        _conv_feat = self.convnet(_imgs)
        _convnet_output_size = _conv_feat.shape # [1, C, H, W]
        spatial_scale = _convnet_output_size[2]/img_H

        self.roi_pool = torchvision.ops.RoIPool(roi_output_size, spatial_scale)

        self.n_visual_feat = _convnet_output_size[1] * roi_output_size[0] * roi_output_size[1]
        self.n_bbox_feat = self.bbox_hidden_dim if self.use_bbox_feat else 0 # x, y, w, h, asp_rat of BBox projected to n_bbox_feat dims
        self.n_own_feat = self.n_visual_feat + self.n_bbox_feat
        self.n_context_feat = self.n_own_feat if self.use_context else 0
        self.n_total_feat = self.n_own_feat + self.n_context_feat

        if self.use_bbox_feat:
            self.bbox_feat_encoder = nn.Sequential(
                nn.Linear(5, self.n_bbox_feat),
                nn.BatchNorm1d(self.n_bbox_feat),
                nn.ReLU(),
                # nn.Linear(self.n_bbox_feat, self.n_bbox_feat),
            )

        if self.use_context and self.use_attention:
            self.q_encoders = nn.ModuleList([nn.Linear(self.n_own_feat, self.hidden_dim) for _ in range(self.n_attn_heads)])
            self.k_encoders = nn.ModuleList([nn.Linear(self.n_context_feat, self.hidden_dim) for _ in range(self.n_attn_heads)])
            
            self.attention_layers = nn.ModuleList([nn.Linear(2*self.hidden_dim, 1) for _ in range(self.n_attn_heads)])
            with torch.no_grad():
                for attn_layer in self.attention_layers:
                    attn_layer.weight.fill_(0)
            
            self.context_combiner = nn.Sequential(
                nn.Linear(self.n_attn_heads * self.n_context_feat, self.n_context_feat),
                nn.BatchNorm1d(self.n_context_feat),
                nn.ReLU(),
            )
        
        self.decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.n_total_feat, self.n_total_feat),
            nn.BatchNorm1d(self.n_total_feat),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.n_total_feat, self.n_classes),
        )

        # print(self)
        print('Model Parameters:', count_parameters(self))
    
    def forward(self, images, bboxes, context_indices):
        """
        Args:
            images: torch.Tensor of size [batch_size, 3, img_H, img_H]
            bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
                each each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            context_indices: [total_n_bboxes_in_batch, n_context] i.e. bbox indices (0-indexed) of contexts for all n bboxes.
                If not enough found, rest are -1
        
        Returns:
            prediction_scores: torch.Tensor of size [total_n_bboxes_in_batch, n_classes]
        """
        batch_size = bboxes.shape[0]

        ##### BBOX FEATURES #####
        if self.use_bbox_feat:
            bbox_features = bboxes[:, 1:].clone() # discard batch_img_index column
            bbox_features[:, 2:] -= bbox_features[:, :2] # convert to [top_left_x, top_left_y, width, height]
            
            bbox_asp_ratio = (bbox_features[:, 2]/bbox_features[:, 3]).view(batch_size, 1)
            bbox_features = torch.cat((bbox_features, bbox_asp_ratio), dim=1)
            
            bbox_features = self.bbox_feat_encoder(bbox_features)
        else:
            bbox_features = bboxes[:, :0] # size [n_bboxes, 0]
        
        ##### OWN VISUAL + BBOX FEATURES #####
        own_features = self.roi_pool(self.convnet(images), bboxes).view(batch_size, self.n_visual_feat)
        own_features = torch.cat((own_features, bbox_features), dim=1)

        ##### CONTEXT VISUAL + BBOX FEATURES USING SELF-ATTENTION #####
        if self.use_context:
            n_context = context_indices.shape[1]
            zero_feat = torch.zeros((1, self.n_own_feat)).to(images.device) # for -1 contexts i.e. extra padded
            own_feat_padded = torch.cat((own_features, zero_feat), dim=0)
            value = own_feat_padded[context_indices.view(-1)].view(batch_size, n_context, self.n_context_feat) # context_features

            if self.use_attention:
                all_head_attention_wts = []
                all_head_context_representations = []
                for head in range(self.n_attn_heads):
                    query = self.q_encoders[head](own_features) # [batch_size, hidden_dim]

                    head_attention_wts = []
                    for c in range(n_context):
                        key = self.k_encoders[head](value[:, c, :]) # [batch_size, hidden_dim]
                        curr_attention_wt = self.attention_layers[head](torch.cat((query, key), dim=1)) # [batch_size, 1]
                        head_attention_wts.append(curr_attention_wt)

                    head_attention_wts = torch.softmax(torch.cat(head_attention_wts, dim=1), dim=1) # [batch_size, n_context]
                    all_head_attention_wts.append(head_attention_wts)
                    
                    head_context_representation = (head_attention_wts.unsqueeze(-1) * value).sum(1) # weighted avg of context bboxes [batch_size, n_context_feat]
                    all_head_context_representations.append(head_context_representation)
                
                multihead_context_representation = torch.cat(all_head_context_representations, dim=1)
                context_representation = self.context_combiner(multihead_context_representation) # [batch_size, n_context_feat]
            else: # average of context features for context representation
                context_representation = value.sum(1) / (context_indices != -1).sum(1).view(batch_size, 1) # [batch_size, n_context_feat]
        else:
            context_representation = own_features[:, :0] # size [n_bboxes, 0]

        ##### FINAL FEATURE VECTOR #####
        combined_feat = torch.cat((own_features, context_representation), dim=1)
        output = self.decoder(combined_feat)

        return output