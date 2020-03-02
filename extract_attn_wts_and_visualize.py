import numpy as np
import os
import random
import torch

from datasets import WebDataset, custom_collate_fn
from models import WebObjExtractionNet
from utils import visualize_bbox


DEVICE_NO = 0
device = torch.device('cuda:%d' % DEVICE_NO if torch.cuda.is_available() else 'cpu')

########## PARAMETERS ##########
N_CLASSES = 4
CLASS_NAMES = ['BG', 'Price', 'Title', 'Image']
IMG_HEIGHT = 1280 # Image assumed to have same height and width

DATA_DIR = '../data/v3/' # Contains imgs/*.png and bboxes/*.pkl files
OUTPUT_DIR = 'results_5-Fold_CV/'

CV_FOLD = 1
SPLIT_DIR = 'splits/'
FOLD_DIR = '%s/Fold-%d' % (SPLIT_DIR, CV_FOLD)
test_img_ids = np.loadtxt('%s/test_imgs.txt' % FOLD_DIR, np.int32)

BACKBONE = 'resnet'
TRAINABLE_CONVNET = True
LEARNING_RATE = 5e-4
BATCH_SIZE = 5
USE_CONTEXT = True # this should be True
CONTEXT_SIZE = 8
USE_ATTENTION = True # this should be True
ATTENTION_HEADS = 1 # if more than 1 attention head, then use score from 1st (index 0) head for visualization
HIDDEN_DIM = 384
ROI_OUTPUT = (3, 3)
USE_BBOX_FEAT = True
BBOX_HIDDEN_DIM = 32
WEIGHT_DECAY = 1e-3
DROP_PROB = 0.2
SAMPLING_FRACTION = 0.8

params = '%s lr-%.0e batch-%d c-%d cs-%d att-%d atth-%d hd-%d roi-%d bbf-%d bbhd-%d wd-%.0e dp-%.1f sf-%.1f' % (BACKBONE, LEARNING_RATE, BATCH_SIZE,
    USE_CONTEXT, CONTEXT_SIZE, USE_ATTENTION, ATTENTION_HEADS, HIDDEN_DIM, ROI_OUTPUT[0], USE_BBOX_FEAT, BBOX_HIDDEN_DIM, WEIGHT_DECAY, DROP_PROB,
    SAMPLING_FRACTION)
results_dir = '%s/%s' % (OUTPUT_DIR, params)
model_save_file = '%s/Fold-%s saved_model.pth' % (results_dir, CV_FOLD)

attention_vis_output_dir = '%s/Fold-%d attention visualization/' % (results_dir, CV_FOLD)
if not os.path.exists(attention_vis_output_dir):
    os.makedirs(attention_vis_output_dir)

########## DATA LOADERS ##########
dataset = WebDataset(DATA_DIR, test_img_ids, USE_CONTEXT, CONTEXT_SIZE, sampling_fraction=1)
model = WebObjExtractionNet(ROI_OUTPUT, IMG_HEIGHT, N_CLASSES, BACKBONE, USE_CONTEXT, USE_ATTENTION, ATTENTION_HEADS, HIDDEN_DIM, USE_BBOX_FEAT,
                            BBOX_HIDDEN_DIM, TRAINABLE_CONVNET, DROP_PROB, CLASS_NAMES).to(device)
model.load_state_dict(torch.load(model_save_file, map_location=device))
model.eval()

for index, img_id in enumerate(test_img_ids):
    print(img_id)
    
    _, images, bboxes, context_indices, labels = custom_collate_fn([dataset.__getitem__(index)])

    images = images.to(device) # [batch_size, 3, img_H, img_W]
    bboxes = bboxes.to(device) # [total_n_bboxes_in_batch, 5]
    context_indices = context_indices.to(device) # [total_n_bboxes_in_batch, 2 * context_size]
    labels = labels.to(device) # [total_n_bboxes_in_batch]
    
    batch_size = bboxes.shape[0]
    with torch.no_grad():
        ##### BBOX FEATURES #####
        bbox_coords = bboxes[:, 1:].clone() # discard batch_img_index column
        bbox_coords[:, 2:] -= bbox_coords[:, :2] # convert to [top_left_x, top_left_y, width, height]

        zero_bbox_coords = torch.zeros(4).view(1,-1).to(device)
        bbox_coords_padded = torch.cat((bbox_coords, zero_bbox_coords), dim=0)
        context_bbox_coords = bbox_coords_padded[context_indices.view(-1)].view(batch_size,-1) # [batch_size, 2 * context_size * 4]

        if USE_BBOX_FEAT:
            bbox_asp_ratio = (bbox_coords[:, 2]/bbox_coords[:, 3]).view(batch_size, 1)
            bbox_features = torch.cat((bbox_coords, bbox_asp_ratio), dim=1)
            
            bbox_features = model.bbox_feat_encoder(bbox_features)
        else:
            bbox_features = bboxes[:, :0] # size [n_bboxes, 0]
        
        ##### OWN VISUAL + BBOX FEATURES #####
        own_features = model.roi_pool(model.convnet(images), bboxes).view(batch_size, model.n_visual_feat)
        own_features = torch.cat((own_features, bbox_features), dim=1)

        ##### CONTEXT VISUAL + BBOX FEATURES USING SELF-ATTENTION #####
        n_context = context_indices.shape[1]
        zero_feat = torch.zeros((1, model.n_own_feat)).to(images.device) # for -1 contexts i.e. extra padded
        own_feat_padded = torch.cat((own_features, zero_feat), dim=0)
        value = own_feat_padded[context_indices.view(-1)].view(batch_size, n_context, model.n_context_feat) # context_features

        head = 0
        attention_wts = []
        query = model.q_encoders[head](own_features) # [batch_size, hidden_dim]
        for c in range(n_context):
            key = model.k_encoders[head](value[:, c, :]) # [batch_size, hidden_dim]
            curr_attention_wt = model.attention_layers[head](torch.cat((query, key), dim=1)) # [batch_size, 1]
            attention_wts.append(curr_attention_wt)

        attention_wts = torch.softmax(torch.cat(attention_wts, dim=1), dim=1) # [batch_size, n_context]

    bbox_coords = bbox_coords[labels > 0] # [x, y, w, h]
    context_bbox_coords = context_bbox_coords[labels > 0]
    attention_wts = attention_wts[labels > 0]
    labels = labels[labels > 0]

    dump_obj = torch.cat((bbox_coords, labels.float().view(-1,1), context_bbox_coords, attention_wts), dim=1).detach().cpu().numpy()
    np.savetxt('%s/%d.csv' % (attention_vis_output_dir, img_id), dump_obj, delimiter=',', fmt='%.3f')

    visualize_bbox('%s/imgs/%d.png' % (DATA_DIR, img_id), '%s/%d.csv' % (attention_vis_output_dir, img_id), attention_vis_output_dir)

print('Extracted attention visualizations and weights for for all images saved in %s' % (attention_vis_output_dir))
print('Each image has a corresponding csv file that stores 4 cols as bbox coordinates, 1 col is label, 2*context_size*4 cols as context bbox coordinates, 2*context_size attention values that sum to 1')
