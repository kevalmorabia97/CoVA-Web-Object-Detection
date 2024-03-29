import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from constants import Constants
from datasets import WebDataset, custom_collate_fn
from models import CoVA
from utils import visualize_bbox

assert (
    len(sys.argv) == 2
), "Usage: python3 extract_attn_wts_and_visualize.py <cv_fold_number>"
CV_FOLD = int(sys.argv[1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CLASSES = Constants.N_CLASSES
CLASS_NAMES = Constants.CLASS_NAMES
IMG_HEIGHT = Constants.IMG_HEIGHT
DATA_DIR = Constants.DATA_DIR
SPLIT_DIR = Constants.SPLIT_DIR
OUTPUT_DIR = Constants.OUTPUT_DIR

FOLD_DIR = "%s/Fold-%d" % (SPLIT_DIR, CV_FOLD)
if CV_FOLD == -1:
    FOLD_DIR = SPLIT_DIR  # use files from SPLIT_DIR

test_img_ids = np.loadtxt("%s/test_imgs.txt" % FOLD_DIR, str)

# Parameters of model for which visualizations are to be created
LEARNING_RATE = 5e-4
BATCH_SIZE = 5
CONTEXT_SIZE = 12
use_context = CONTEXT_SIZE > 0
HIDDEN_DIM = 384
ROI_OUTPUT = (3, 3)
BBOX_HIDDEN_DIM = 32
USE_ADDITIONAL_FEAT = False
WEIGHT_DECAY = 1e-3
DROP_PROB = 0.2
SAMPLING_FRACTION = 0.9

assert CONTEXT_SIZE > 0, "Attention Scores can only be computed if CONTEXT_SIZE > 0"

params = "lr-%.0e batch-%d cs-%d hd-%d roi-%d bbhd-%d af-%d wd-%.0e dp-%.1f sf-%.1f" % (
    LEARNING_RATE,
    BATCH_SIZE,
    CONTEXT_SIZE,
    HIDDEN_DIM,
    ROI_OUTPUT[0],
    BBOX_HIDDEN_DIM,
    USE_ADDITIONAL_FEAT,
    WEIGHT_DECAY,
    DROP_PROB,
    SAMPLING_FRACTION,
)
results_dir = "%s/%s" % (OUTPUT_DIR, params)
model_save_file = "%s/Fold-%s saved_model.pth" % (results_dir, CV_FOLD)

attention_vis_output_dir = "%s/Fold-%d attention visualization/" % (
    results_dir,
    CV_FOLD,
)
if not os.path.exists(attention_vis_output_dir):
    os.makedirs(attention_vis_output_dir)

########## DATA LOADERS ##########
dataset = WebDataset(
    DATA_DIR, test_img_ids, CONTEXT_SIZE, USE_ADDITIONAL_FEAT, sampling_fraction=1
)
n_additional_feat = dataset.n_additional_feat
model = CoVA(
    ROI_OUTPUT,
    IMG_HEIGHT,
    N_CLASSES,
    use_context,
    HIDDEN_DIM,
    BBOX_HIDDEN_DIM,
    n_additional_feat,
    DROP_PROB,
    CLASS_NAMES,
).to(device)
model.load_state_dict(torch.load(model_save_file, map_location=device))
model.eval()

for index, img_id in tqdm(enumerate(test_img_ids), total=len(test_img_ids)):
    _, images, bboxes, additional_feats, context_indices, labels = custom_collate_fn(
        [dataset.__getitem__(index)]
    )

    images = images.to(device)  # [batch_size, 3, img_H, img_W]
    bboxes = bboxes.to(device)  # [total_n_bboxes_in_batch, 5]
    additional_feats = additional_feats.to(
        device
    )  # [total_n_bboxes_in_batch, n_additional_feat]
    context_indices = context_indices.to(
        device
    )  # [total_n_bboxes_in_batch, 2 * context_size]
    labels = labels.to(device)  # [total_n_bboxes_in_batch]

    N = bboxes.shape[0]
    with torch.no_grad():
        bbox_coords = bboxes[:, 1:].clone()  # discard batch_img_index column
        bbox_coords[:, 2:] -= bbox_coords[
            :, :2
        ]  # convert to [top_left_x, top_left_y, width, height]

        zero_bbox_coords = torch.zeros(4).view(1, -1).to(device)
        bbox_coords_padded = torch.cat((bbox_coords, zero_bbox_coords), dim=0)
        context_bbox_coords = bbox_coords_padded[context_indices.view(-1)].view(
            N, -1
        )  # [N, 2 * context_size * 4]

        visual_feats = model._get_visual_features(images, bboxes)
        bbox_feats = model._get_bbox_features(bboxes)
        additional_feats = model.bn_additional_feat(additional_feats)
        own_features = torch.cat((visual_feats, bbox_feats, additional_feats), dim=1)

        _, attention_wts = model.gat(
            own_features, context_indices, return_attn_wts=True
        )

    bbox_coords = bbox_coords[labels > 0]  # [x, y, w, h]
    context_bbox_coords = context_bbox_coords[labels > 0]
    attention_wts = attention_wts[labels > 0]
    labels = labels[labels > 0]

    dump_obj = (
        torch.cat(
            (
                bbox_coords,
                labels.float().view(-1, 1),
                context_bbox_coords,
                attention_wts,
            ),
            dim=1,
        )
        .detach()
        .cpu()
        .numpy()
    )
    np.savetxt(
        "%s/%s.csv" % (attention_vis_output_dir, img_id),
        dump_obj,
        delimiter=",",
        fmt="%.3f",
    )

    visualize_bbox(
        "%s/imgs/%s.png" % (DATA_DIR, img_id),
        "%s/%s.csv" % (attention_vis_output_dir, img_id),
        attention_vis_output_dir,
    )

print(
    "Extracted attention visualizations and weights for for all images saved in %s"
    % (attention_vis_output_dir)
)
print(
    "Each image has a corresponding csv file that stores 4 cols as bbox coordinates, 1 col is label, 2*context_size*4 cols as context bbox coordinates, 2*context_size attention values that sum to 1"
)
