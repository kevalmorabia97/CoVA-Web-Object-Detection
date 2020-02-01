import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision

from utils import pkl_load


class WebDataset(torchvision.datasets.VisionDataset):
    """
    Class to load train/val/test datasets
    """
    def __init__(self, root, img_ids, context_size, max_bg_boxes=-1):
        """
        Args:
            root: directory where data is located
                Must contain imgs/x.png Image and corresponding bboxes/x.pkl BBox coordinates file 
                BBox file should have bboxes in pre-order and each row corresponds to [x,y,w,h,label]
            img_ids: list of img_names to consider
            context_size: number of BBoxes before and after to consider as context
            max_bg_boxes: randomly sample this many number of background boxes (class 0) while training (default: -1 --> no sampling, take all)
                All samples of class > 0 are always taken
                NOTE: For val and test data, max_bg_boxes SHOULD be -1 (no sampling)
        """
        super(WebDataset, self).__init__(root)
        
        self.ids = img_ids
        self.context_size = context_size
        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([0.8992, 0.8977, 0.8966], [0.2207, 0.2166, 0.2217]) # calculated on trainval data
        ])
        self.max_bg_boxes = max_bg_boxes
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index in range [0, self.__len__ - 1]

        Returns:
            image: torch.Tensor of size [3,H,W].
            bboxes: torch.Tensor of size [n_bbox, 4] i.e. n bboxes each of [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            context_indices: torch.Tensor of size [n_bbox, 2*context_size] i.e. bbox indices (0-indexed) of contexts for all n bboxes.
                If not enough found, rest are -1
            labels: torch.Tensor of size [n_bbox] i.e. each value is label of the corresponding bbox
        """
        img_id = self.ids[index]
        
        img = Image.open('%s/imgs/%s.png' % (self.root, img_id)).convert('RGB')
        img = self.img_transform(img)
        
        bboxes = pkl_load('%s/bboxes/%s.pkl' % (self.root, img_id))
        if self.max_bg_boxes > 0:
            ## TODO: Make sure order is preserved
            bg_boxes = bboxes[bboxes[:,-1] == 0]
            pos_boxes = bboxes[bboxes[:,-1] != 0]

            indices = np.random.permutation(len(bg_boxes))[:self.max_bg_boxes]
            bg_boxes = bg_boxes[indices]

            bboxes = np.concatenate((pos_boxes, bg_boxes), axis=0)
        
        labels = torch.LongTensor(bboxes[:,-1])

        bboxes = torch.Tensor(bboxes[:,:-1])
        bboxes[:,2:] += bboxes[:,:2] # convert from [x,y,w,h] to [x1,y1,x2,y2]

        context_indices = []
        for i in range(bboxes.shape[0]):
            context = list(range(max(0, i-self.context_size), i)) + list(range(i+1, min(bboxes.shape[0], i+self.context_size+1)))
            context_indices.append(context + [-1]*(2*self.context_size - len(context)))
        context_indices = torch.LongTensor(context_indices)

        return img, bboxes, context_indices, labels

    def __len__(self):
        return len(self.ids)

########################## End of class `WebDataset` ##########################


def custom_collate_fn(batch):
    """
    Since all images might have different number of BBoxes, to use batch_size > 1,
    custom collate_fn has to be created that creates a batch
    
    Args:
        batch: list of N=`batch_size` tuples. Example [(img_1, bboxes_1, ci_1, labels_1), ..., (img_N, bboxes_N, ci_N, labels_N)]
    
    Returns:
        batch: contains images, bboxes, labels
            images: torch.Tensor [N, 3, img_H, img_W]
            bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
                each each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            context_indices: [total_n_bboxes_in_batch, 2*context_size] i.e. bbox indices (0-indexed) of contexts for all n bboxes.
                If not enough found, rest are -1
            
            labels: torch.Tensor [total_n_bboxes_in_batch]
    """
    images, bboxes, context_indices, labels = zip(*batch)
    # images = (img_1, ..., img_N) each element of size [3, img_H, img_W]
    # bboxes = (bboxes_1, ..., bboxes_N) each element of size [n_bboxes_in_image, 4]
    # context_indices = (ci_1, ..., ci_N) each element of size [n_bboxes_in_image, 2*context_size]
    # labels = (labels_1, ..., labels_N) each element of size [n_bboxes_in_image]
    
    images = torch.stack(images, 0)
    
    bboxes_with_batch_index = []
    observed_bboxes = 0
    for i, bbox in enumerate(bboxes):
        batch_indices = torch.Tensor([i]*bbox.shape[0]).view(-1,1)
        bboxes_with_batch_index.append(torch.cat((batch_indices, bbox), dim=1))
        context_indices[i][context_indices[i] != -1] += observed_bboxes
        observed_bboxes += bbox.shape[0]
    bboxes_with_batch_index = torch.cat(bboxes_with_batch_index)
    context_indices = torch.cat(context_indices)
    
    labels = torch.cat(labels)
    
    return images, bboxes_with_batch_index, context_indices, labels


def load_data(data_dir, train_img_ids, val_img_ids, test_img_ids, context_size, batch_size, num_workers=4, max_bg_boxes=-1):
    """
    Args:
        data_dir: directory which contains x.png Image and corresponding x.pkl BBox coordinates file
        train_img_ids: list of img_names to consider in train split
        val_img_ids: list of img_names to consider in val split
        test_img_ids: list of img_names to consider in test split
        context_size: number of BBoxes before and after to consider as context
        batch_size: size of batch in train_loader
        max_bg_boxes: randomly sample this many number of background boxes (class 0) while training (default: -1 --> no sampling, take all)
            All samples of class > 0 are always taken
            NOTE: For val and test data, max_bg_boxes SHOULD be -1 (no sampling)
    
    Returns:
        train_loader, val_loader, test_loader (torch.utils.data.DataLoader)
    """
    assert np.intersect1d(train_img_ids, val_img_ids).size == 0
    assert np.intersect1d(val_img_ids, test_img_ids).size == 0
    assert np.intersect1d(train_img_ids, test_img_ids).size == 0
    
    train_dataset = WebDataset(data_dir, train_img_ids, context_size, max_bg_boxes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=custom_collate_fn, drop_last=False)

    val_dataset = WebDataset(data_dir, val_img_ids, context_size, max_bg_boxes=-1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                            collate_fn=custom_collate_fn, drop_last=False)
    
    test_dataset = WebDataset(data_dir, test_img_ids, context_size, max_bg_boxes=-1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                             collate_fn=custom_collate_fn, drop_last=False)
    
    print('---> No. of Images\t Train: %d\t Val: %d\t Test: %d\n' % ( len(train_dataset), len(val_dataset), len(test_dataset) ))
    
    return train_loader, val_loader, test_loader