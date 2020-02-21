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
USE_CONTEXT = True
CONTEXT_SIZE = 6
USE_ATTENTION = True
HIDDEN_DIM = 300
ROI_OUTPUT = (3, 3)
USE_BBOX_FEAT = True
WEIGHT_DECAY = 1e-3
DROP_PROB = 0.2
MAX_BG_BOXES = 100

params = '%s lr-%.0e batch-%d c-%d cs-%d att-%d hd-%d roi-%d bbf-%d wd-%.0e dp-%.2f mbb-%d' % (BACKBONE, LEARNING_RATE, BATCH_SIZE,
    USE_CONTEXT, CONTEXT_SIZE, USE_ATTENTION, HIDDEN_DIM, ROI_OUTPUT[0], USE_BBOX_FEAT, WEIGHT_DECAY, DROP_PROB, MAX_BG_BOXES)
results_dir = '%s/%s' % (OUTPUT_DIR, params)
model_save_file = '%s/Fold-%s saved_model.pth' % (results_dir, CV_FOLD)

attention_vis_output_dir = '%s/Fold-%d attention visualization/' % (results_dir, CV_FOLD)
if not os.path.exists(attention_vis_output_dir):
    os.makedirs(attention_vis_output_dir)

########## DATA LOADERS ##########
dataset = WebDataset(DATA_DIR, test_img_ids, USE_CONTEXT, CONTEXT_SIZE, max_bg_boxes=-1)
model = WebObjExtractionNet(ROI_OUTPUT, IMG_HEIGHT, N_CLASSES, BACKBONE, USE_CONTEXT, USE_ATTENTION, HIDDEN_DIM, TRAINABLE_CONVNET,
                            DROP_PROB, USE_BBOX_FEAT, CLASS_NAMES).to(device)
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
        ##### OWN VISUAL FEATURES #####
        conv_feat = model.convnet(images)
        own_features = model.roi_pool(conv_feat, bboxes).view(batch_size,-1)

        ##### CONTEXT VISUAL FEATURES USING ATTENTION #####
        zero_feat = torch.zeros(model.n_visual_feat).view(1,-1).to(device) # for -1 contexts i.e. extra padded
        pooled_feat_padded = torch.cat((own_features, zero_feat), dim=0)
        context_features = pooled_feat_padded[context_indices.view(-1)].view(batch_size,-1) # [batch_size, 2 * context_size * n_visual_feat]

        own_features_encoded = model.encoder(own_features)
        attention_wts = []
        for i in range(2*CONTEXT_SIZE):
            curr_features_encoded = model.encoder(context_features[:, i*model.n_visual_feat:(i+1)*model.n_visual_feat])
            concatenated_feat = torch.cat((own_features_encoded, curr_features_encoded), dim=1)
            curr_attention_wt = model.attention_layer(concatenated_feat) # [batch_size, 1]
            attention_wts.append(curr_attention_wt)
        attention_wts = torch.softmax(torch.cat(attention_wts, dim=1), dim=1) # [batch_size, 2 * context_size, 1]
    
    bbox_features = bboxes[:, 1:].clone() # remove batch_index column
    bbox_features[:, 2:] -= bbox_features[:, :2] # convert to [top_left_x, top_left_y, width, height]
    
    zero_bbox_feat = torch.zeros(4).view(1,-1).to(device)
    bbox_features_padded = torch.cat((bbox_features, zero_bbox_feat), dim=0)
    context_bbox_features = bbox_features_padded[context_indices.view(-1)].view(batch_size,-1) # [batch_size, 2 * context_size * 4]

    bbox_features = bbox_features[labels > 0]
    context_bbox_features = context_bbox_features[labels > 0]
    attention_wts = attention_wts[labels > 0]
    labels = labels[labels > 0]

    dump_obj = torch.cat((bbox_features, labels.float().view(-1,1), context_bbox_features, attention_wts), dim=1).detach().cpu().numpy()
    np.savetxt('%s/%d.csv' % (attention_vis_output_dir, img_id), dump_obj, delimiter=',', fmt='%.3f')

    visualize_bbox('%s/imgs/%d.png' % (DATA_DIR, img_id), '%s/%d.csv' % (attention_vis_output_dir, img_id), attention_vis_output_dir)

print('Extracted attention visualizations and weights for for all images saved in %s' % (attention_vis_output_dir))
print('Each image has a corresponding csv file that stores 4 cols as bbox coordinates, 1 col is label, 2*context_size*4 cols as context bbox coordinates, 2*context_size attention values that sum to 1')
