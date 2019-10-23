import numpy as np
import os
from PIL import Image
import torch
import torchvision
from torchvision import transforms

from utils import pkl_load


class WebDataset(torchvision.datasets.VisionDataset):
    """
    Class to load train/val/test datasets
    """
    def __init__(self, root):
        """
        Args:
            root: directory where data is located
            Must contain x.png Image and corresponding x.pkl BBox coordinates file
        """
        super(WebDataset, self).__init__(root)
        
        self.ids = [f.name.split('.')[0] for f in os.scandir(self.root) if f.is_file() and 'png' in f.name]
        self.img_transform = transforms.ToTensor()
        ## convert to 0 MEAN, 1 VAR ???
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index in range [0, self.__len__ - 1]

        Returns:
            image: torch.Tensor of size [3,H,W].
            bboxes: torch.Tensor of size [n_bbox, 4] i.e. n bboxes each of [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            labels: torch.Tensor of size [n_bbox] i.e. each value is label of the corresponding bbox
        """
        img_id = self.ids[index]
        
        img = Image.open('%s/%s.png' % (self.root, img_id)).convert('RGB')
        img = self.img_transform(img)
        
        input_boxes = pkl_load('%s/%s.pkl' % (self.root, img_id))
        bboxes = torch.Tensor( np.concatenate((input_boxes['gt_boxes'], input_boxes['other_boxes']), axis=0) )
        bboxes[:,2:] += bboxes[:,:2]
        
        labels = torch.Tensor([1,2,3] + [0]*input_boxes['other_boxes'].shape[0]).long()

        return img, bboxes, labels

    def __len__(self):
        return len(self.ids)