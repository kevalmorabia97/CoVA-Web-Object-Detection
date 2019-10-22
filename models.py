import torch
import torch.nn as nn
import torchvision


class WebIENet(nn.Module):
    def __init__(self, roi_output_size, img_H, n_classes, learnable_convnet=False):
        """
        Args:
            roi_output_size: Tuple (int, int) which will be output of the roi_pool layer for each channel of convnet_feature
            img_H: height of image given as input to the convnet. Image assumed to be of same W and H
            n_classes: num of classes for BBoxes
            learnable_convnet: if True then convnet weights will be modified while training (default: False)
        """
        print('Initializing WebIENet model...')
        super(WebIENet, self).__init__()
        self.learnable_convnet = learnable_convnet
        
        print('Using first few layers of Resnet18 as Image Feature Extractor')
        self.convnet = torchvision.models.resnet18(pretrained=True)
        modules = list(self.convnet.children())[:-5] # remove last few layers!
        self.convnet = nn.Sequential(*modules)
        
        if self.learnable_convnet == False:
            for p in self.convnet.parameters(): # freeze weights
                p.requires_grad = False
        
        _imgs = torch.autograd.Variable(torch.Tensor(1, 3, img_H, img_H))
        _conv_feat = self.convnet(_imgs)
        _convnet_output_size = _conv_feat.size() # [1, C, H, W]
        n_feat = _convnet_output_size[1] * roi_output_size[0] * roi_output_size[1]
        
        spatial_scale = _convnet_output_size[2]/img_H
        self.roi_pool = torchvision.ops.RoIPool(roi_output_size, spatial_scale)
        self.fc = nn.Linear(n_feat, n_classes)
        
        print('Input batch of images:', _imgs.size())
        print('ConvNet feature:', _convnet_output_size)
        print('RoI Pooling Spatial Scale:', spatial_scale)
        print('Classifier 1st FC layer input features:', n_feat)
        print('-'*50)
    
    def forward(self, images, bboxes):
        """
        Args:
            images: torch.Tensor of size [batch_size, 3, img_H, img_H]
            bboxes: list of torch.Tensor, each of size [n_bboxes, 4]
                each of the 4 values correspond to [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        
        TODO: CURRENTLY WORKS ONLY FOR BATCHSIZE=1 :-(
        
        Returns:
            prediction_scores: torch.Tensor of size [n_bboxes, n_classes]
        """
        conv_feat = self.convnet(images)
        pooled_feat = self.roi_pool(conv_feat, bboxes)
        pooled_feat = pooled_feat.view(pooled_feat.size()[0],-1)
        output = self.fc(pooled_feat)

        return output