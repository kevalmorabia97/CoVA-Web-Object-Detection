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
    def __init__(self, root, img_ids, context_size, use_additional_feats=False, sampling_fraction=1):
        """
        Args:
            root: directory where data is located
                Must contain imgs/x.png Image and corresponding bboxes/x.pkl Bounding Boxes files
                BBox file should have bboxes in pre-order and each row corresponds to [x,y,w,h,label]
            img_ids: list of img_names to consider
            context_size: number of BBoxes before and after to consider as context / graph neighborhood (int)
                If set to 0, `context_indices` will be empty as it will not be used
            use_additional_feats: whether to use additional features (default: False)
                if True, `root` directory must contain additional_features/x.pkl additional features which is a numpy array of shape [n_bboxes, n_additional_feat]
            sampling_fraction: randomly sample this many (float between 0 and 1) fraction of background boxes (class 0) while training (default: 1 --> no sampling, take all)
                All samples of class > 0 are always taken
                NOTE: For val and test data, sampling_fraction SHOULD be 1 (no sampling)
        """
        super(WebDataset, self).__init__(root)
        assert context_size >= 0
        assert sampling_fraction > 0 and sampling_fraction <= 1
        
        self.ids = img_ids
        self.context_size = context_size
        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        self.sampling_fraction = sampling_fraction

        self.imgs_paths = ['%s/imgs/%s.png' % (self.root, img_id) for img_id in self.ids]
        self.all_bboxes = [pkl_load('%s/bboxes/%s.pkl' % (self.root, img_id)) for img_id in self.ids]
        
        if use_additional_feats:
            self.all_additional_tensor_features = [torch.Tensor(pkl_load('%s/additional_features/%s.pkl' % (self.root, img_id))) for img_id in self.ids]
            self.n_additional_feat = len(self.all_additional_tensor_features[0][0])
        else:
            self.all_additional_tensor_features = [torch.empty(len(bboxes), 0) for bboxes in self.all_bboxes]
            self.n_additional_feat = 0
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index in range [0, self.__len__ - 1]

        Returns:
            img_id: name of image (string)
            image: torch.Tensor of size [3,H,W].
            bboxes: torch.Tensor of size [n_bbox, 4] i.e. n bboxes each of [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            additional_feats: torch.Tensor of size [n_bbox, n_additional_feat]
            context_indices: torch.LongTensor of size [n_bbox, 2*context_size] i.e. bbox indices (0-indexed) of contexts for all n bboxes.
                If not enough found, rest are -1
            labels: torch.LongTensor of size [n_bbox] i.e. each value is label of the corresponding bbox
        """
        img_id = self.ids[index]
        
        img = Image.open(self.imgs_paths[index]).convert('RGB')
        img = self.img_transform(img)
        
        bboxes = self.all_bboxes[index]
        additional_feats = self.all_additional_tensor_features[index]
        if self.sampling_fraction < 1: # preserve order, include all non-BG bboxes
            sampled_bbox_idxs = np.random.permutation(bboxes.shape[0])[:int(self.sampling_fraction * bboxes.shape[0])]
            indices = np.concatenate((np.where(bboxes[:,-1] != 0)[0], sampled_bbox_idxs))
            indices = np.unique(indices) # sort and remove duplicate non-BG boxes
            bboxes = bboxes[indices]
            additional_feats = additional_feats[indices]
        
        labels = torch.LongTensor(bboxes[:,-1])

        bboxes = torch.Tensor(bboxes[:,:-1])
        bboxes[:,2:] += bboxes[:,:2] # convert from [x,y,w,h] to [x1,y1,x2,y2]

        if self.context_size > 0: # Neighborhood consists of `context_size` elements on both sides in preorder traversal
            context_indices = []
            for i in range(bboxes.shape[0]):
                context = list(range(max(0, i-self.context_size), i)) + list(range(i+1, min(bboxes.shape[0], i+self.context_size+1)))
                context_indices.append(context + [-1]*(2*self.context_size - len(context)))
            context_indices = torch.LongTensor(context_indices)
        else:
            context_indices = torch.empty((0, 0), dtype=torch.long)

        return img_id, img, bboxes, additional_feats, context_indices, labels

    def __len__(self):
        return len(self.ids)

########################## End of class `WebDataset` ##########################


def custom_collate_fn(batch):
    """
    Since all images might have different number of BBoxes, to use batch_size > 1,
    custom collate_fn has to be created that creates a batch
    
    Args:
        batch: list of N=`batch_size` tuples. Example [(img_id_1, img_1, bboxes_1, afs_1, ci_1, labels_1), ..., (img_id_N, img_N, bboxes_N, afs_N, ci_N, labels_N)]
    
    Returns:
        batch: contains img_ids, images, bboxes, context_indices, labels
            img_ids: names of images (string) to compute imgwise (webpagewise) and domainwise (macro) Accuracies
            images: torch.Tensor [N, 3, img_H, img_W]
            bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
                each each of [batch_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            additional_feats: torch.Tensor of size [total_n_bboxes_in_batch, n_additional_feat]
            context_indices: torch.LongTensor [total_n_bboxes_in_batch, 2*context_size] i.e. bbox indices (0-indexed) of contexts for all n bboxes.
                If not enough found, rest are -1
            labels: torch.LongTensor [total_n_bboxes_in_batch]
    """
    img_ids, images, bboxes, additional_feats, context_indices, labels = zip(*batch)
    # img_ids = (img_id_1, ..., img_id_N)
    # images = (img_1, ..., img_N) each element of size [3, img_H, img_W]
    # bboxes = (bboxes_1, ..., bboxes_N) each element of size [n_bboxes_in_image, 4]
    # additional_feats = (additional_feats_1, ..., additional_feats_N) each element of size [n_bboxes_in_image, n_additional_feat]
    # context_indices = (ci_1, ..., ci_N) each element of size [n_bboxes_in_image, 2*context_size]
    # labels = (labels_1, ..., labels_N) each element of size [n_bboxes_in_image]
    
    img_ids = np.array(img_ids)
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

    additional_feats = torch.cat(additional_feats)
    labels = torch.cat(labels)
    
    return img_ids, images, bboxes_with_batch_index, additional_feats, context_indices, labels


def load_data(data_dir, train_img_ids, val_img_ids, test_img_ids, context_size, batch_size, use_additional_feats=False, sampling_fraction=1, num_workers=4):
    """
    Args:
        data_dir: directory which contains imgs/x.png Image and corresponding bboxes/x.pkl BBox coordinates file
        train_img_ids: list of img_names to consider in train split
        val_img_ids: list of img_names to consider in val split
        test_img_ids: list of img_names to consider in test split
        context_size: number of BBoxes before and after to consider as context
        batch_size: size of batch in train_loader
        use_additional_feats: whether to use additional features (default: False)
            if True, `root` directory must contain additional_feats/x.pkl additional features which is a numpy array of shape [n_bboxes, n_additional_feat]
        sampling_fraction: randomly sample this many fraction of background boxes (class 0) while training (default: 1 --> no sampling, take all)
            All samples of class > 0 are always taken
    
    Returns:
        train_loader, val_loader, test_loader (torch.utils.data.DataLoader)
    """
    assert np.intersect1d(train_img_ids, val_img_ids).size == 0
    assert np.intersect1d(val_img_ids, test_img_ids).size == 0
    assert np.intersect1d(train_img_ids, test_img_ids).size == 0
    
    train_dataset = WebDataset(data_dir, train_img_ids, context_size, use_additional_feats, sampling_fraction)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=custom_collate_fn, drop_last=False)

    val_dataset = WebDataset(data_dir, val_img_ids, context_size, use_additional_feats, sampling_fraction=1)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=num_workers,
                            collate_fn=custom_collate_fn, drop_last=False)
    
    test_dataset = WebDataset(data_dir, test_img_ids, context_size, use_additional_feats, sampling_fraction=1)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=num_workers,
                             collate_fn=custom_collate_fn, drop_last=False)
    
    print('No. of Images\t Train: %d\t Val: %d\t Test: %d\n' % ( len(train_dataset), len(val_dataset), len(test_dataset) ))
    
    return train_loader, val_loader, test_loader