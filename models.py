import numpy as np
import torch
import torch.nn as nn
import torchvision

from utils import count_parameters


class CoVA(nn.Module):
    def __init__(self, roi_output_size, img_H, n_classes, use_context=True, hidden_dim=384, bbox_hidden_dim=32, 
                 n_additional_feat=0, drop_prob=0.2, class_names=None):
        """
        Implementation of CoVA: Context-aware Visual Attention for Webpage Information Extraction

        roi_output_size: Tuple (int, int) which will be output of the roi_pool layer for each channel of convnet_feature
        img_H: height of image given as input to the convnet. Image assumed to be of same W and H
        n_classes: num of classes for BBoxes
        use_context: if True, use context for context_representation (using GAT) along with h_i (default: True) 
        hidden_dim: size of hidden contextual representation, used when use_context=True (default: 384)
        bbox_hidden_dim: if > 0, size of hidden representation of [x,y,w,h,asp_ratio] bbox features (default: 32)
        n_additional_feat: num of additional features for each bbox to be used along with visual and bbox features
        drop_prob: dropout probability (default: 0.2)
        class_names: list of n_classes string elements containing names of the classes (default: [0, 1, ..., n_classes-1])
        """
        super(CoVA, self).__init__()

        self.n_classes = n_classes
        self.use_context = use_context
        self.hidden_dim = hidden_dim
        self.bbox_hidden_dim = bbox_hidden_dim
        self.n_additional_feat = n_additional_feat
        self.class_names = np.arange(self.n_classes).astype(str) if class_names is None else class_names

        ##### REPRESENTATION NETWORK (RN) #####
        self.convnet = torchvision.models.resnet18(pretrained=True)
        modules = list(self.convnet.children())[:-5] # remove last few layers!
        self.convnet = nn.Sequential(*modules)

        _imgs = torch.autograd.Variable(torch.Tensor(1, 3, img_H, img_H))
        _conv_feat = self.convnet(_imgs)
        _convnet_output_size = _conv_feat.shape # [1, C, H, W]
        spatial_scale = _convnet_output_size[2]/img_H

        self.roi_pool = torchvision.ops.RoIPool(roi_output_size, spatial_scale)

        self.n_visual_feat = _convnet_output_size[1] * roi_output_size[0] * roi_output_size[1]
        self.n_feat = self.n_visual_feat + self.bbox_hidden_dim + self.n_additional_feat

        if self.bbox_hidden_dim > 0:
            self.bbox_feat_encoder = nn.Sequential(
                nn.Linear(5, self.bbox_hidden_dim),
                nn.BatchNorm1d(self.bbox_hidden_dim),
                nn.ReLU(),
            )
        
        if self.n_additional_feat > 0:
            self.bn_additional_feat = nn.BatchNorm1d(self.n_additional_feat)
        else:
            self.bn_additional_feat = lambda x: x

        ##### GRAPH ATTENTION LATER (GAT) #####
        if self.use_context:
            self.gat = GraphAttentionLayer(self.n_feat, self.hidden_dim)
        
        ##### FC LAYERS #####
        self.n_total_feat = self.n_feat + self.hidden_dim
        self.decoder = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.n_total_feat, self.n_total_feat),
            nn.BatchNorm1d(self.n_total_feat),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.n_total_feat, self.n_classes),
        )

        print('Model Parameters:', count_parameters(self))
    
    def forward(self, images, bboxes, additional_feats, context_indices):
        """
        images: torch.Tensor of size [batch_size, 3, img_H, img_H]
        bboxes: torch.Tensor [N, 5], N = total_n_bboxes_in_batch
            each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        additional_feats: torch.Tensor [N, n_additional_feat]
        context_indices: Torch.LongTensor [N, n_context]
            indices (0 to N-1) of `n_context` bboxes that are in context for a given bbox. If not enough found, rest are -1
        
        Returns:
            prediction_scores: torch.Tensor of size [N, n_classes]
        """
        N = bboxes.shape[0]

        ##### OWN VISUAL + BBOX FEATURES + ADDITIONAL FEATURES #####
        visual_feats = self._get_visual_features(images, bboxes)
        bbox_feats = self._get_bbox_features(bboxes)
        additional_feats = self.bn_additional_feat(additional_feats)
        own_features = torch.cat((visual_feats, bbox_feats, additional_feats), dim=1)

        ##### CONTEXT FEATURES USING GRAPH ATTENTION LAYER #####
        if self.use_context:
            context_representation = self.gat(own_features, context_indices)
        else:
            context_representation = own_features[:, :0] # size [n_bboxes, 0]

        ##### FINAL FEATURE VECTOR #####
        combined_feat = torch.cat((own_features, context_representation), dim=1)
        output = self.decoder(combined_feat)

        return output

    def _get_visual_features(self, images, bboxes):
        return self.roi_pool(self.convnet(images), bboxes).view(bboxes.shape[0], self.n_visual_feat)

    def _get_bbox_features(self, bboxes):
        """
        Get [x,y,w,h,asp_ratio] for each bbox and transform to n_bbox_feat if bbox fetaures are to be used
        """
        if self.bbox_hidden_dim > 0:
            bbox_feats = bboxes[:, 1:].clone() # discard batch_img_index column
            bbox_feats[:, 2:] -= bbox_feats[:, :2] # convert to [top_left_x, top_left_y, width, height]
            
            bbox_asp_ratio = (bbox_feats[:, 2]/bbox_feats[:, 3]).view(bboxes.shape[0], 1)
            bbox_feats = torch.cat((bbox_feats, bbox_asp_ratio), dim=1)
            
            bbox_feats = self.bbox_feat_encoder(bbox_feats)
        else:
            bbox_feats = bboxes[:, :0] # size [n_bboxes, 0]
        
        return bbox_feats


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, hidden_dim, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        self.W_i = nn.Linear(self.in_features, self.hidden_dim, bias=False)
        self.W_j = nn.Linear(self.in_features, self.hidden_dim, bias=False)
        
        self.attention_layer = nn.Linear(2*self.hidden_dim, 1)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # nn.init.xavier_uniform_(self.W_i.weight, gain=1.414)
        # nn.init.xavier_uniform_(self.W_j.weight, gain=1.414)
        # nn.init.xavier_uniform_(self.attention_layer.weight, gain=1.414)

    def forward(self, h_i, context_indices, return_attn_wts=False):
        """
        h_i: features for all bboxes torch.Tensor of shape [N, in_features]
        context_indices: Torch.LongTensor [N, n_context]
            ids (0 to N-1) of `n_context` bboxes that are in context (neighborhood) for a given bbox.
            If not enough found, rest are -1
        """
        N, n_context = context_indices.shape

        zero_feat = torch.zeros((1, self.in_features)).to(h_i.device) # to map -1 contexts to zero_feat
        h_i_padded = torch.cat((h_i, zero_feat), dim=0)
        h_j = h_i_padded[context_indices.view(-1)].view(N, n_context, self.in_features) # context_features

        Wh_i = self.W_i(h_i) # [N, hidden_dim]
        Wh_i_repeated = Wh_i.repeat_interleave(n_context, dim=0).view(N, n_context, self.hidden_dim)

        Wh_j = self.W_j(h_j) # [N, n_context, hidden_dim]

        attention_wts = self.attention_layer(torch.cat((Wh_i_repeated, Wh_j), dim=2)).squeeze(2) # [N, n_context]
        attention_wts = self.leakyrelu(attention_wts)

        minus_inf = -9e15*torch.ones_like(attention_wts)
        attention_wts = torch.where(context_indices >= 0, attention_wts, minus_inf)
        attention_wts = torch.softmax(attention_wts, dim=1) # [N, n_context]
        
        h_prime = (attention_wts.unsqueeze(-1) * Wh_j).sum(1) # weighted avg of contexts [N, hidden_dim]

        if return_attn_wts:
            return h_prime, attention_wts
        return h_prime