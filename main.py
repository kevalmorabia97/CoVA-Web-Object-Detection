import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_data
from models import WebObjExtractionNet
from train import train_model, evaluate_model
from utils import print_and_log


########## CMDLINE ARGS ##########
parser = argparse.ArgumentParser('Train Model')
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('-e', '--n_epochs', type=int, default=100)
parser.add_argument('-bb', '--backbone', type=str, default='alexnet', choices=['alexnet', 'resnet'])
parser.add_argument('-tc', '--trainable_convnet', type=int, default=1, choices=[0,1])
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005)
parser.add_argument('-bs', '--batch_size', type=int, default=25)
parser.add_argument('-mbb', '--max_bg_boxes', type=int, default=100)
parser.add_argument('-wd', '--weight_decay', type=float, default=0.001)
parser.add_argument('-r', '--roi', type=int, default=1)
parser.add_argument('-dp', '--drop_prob', type=float, default=0.5)
parser.add_argument('-pf', '--pos_feat', type=int, default=1, choices=[0,1])
parser.add_argument('-nw', '--num_workers', type=int, default=4)
parser.add_argument('-s', '--split', type=str, choices=['random', 'domain_wise'])
args = parser.parse_args()

device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')

########## MAKING RESULTS REPRODUCIBLE ##########
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

########## PARAMETERS ##########
N_CLASSES = 4
CLASS_NAMES = ['BG', 'Price', 'Title', 'Image']
IMG_HEIGHT = 1280 # Image assumed to have same height and width
EVAL_INTERVAL = 3 # Number of Epochs after which model is evaluated
NUM_WORKERS = args.num_workers # multithreaded data loading

DATA_DIR = '/shared/data_product_info/v1_6k/' # Contains .png and .pkl files for train and test data
OUTPUT_DIR = 'results' # logs are saved here! 
# NOTE: if same hyperparameter configuration is run again, previous log file and saved model will be overwritten

SPLIT_DIR = 'splits/' + args.split # contains train, val, test split files
# each line in these files should contain name of the training image (without file extension)
TRAIN_SPLIT_ID_FILE = SPLIT_DIR+ '/train_imgs.txt'
VAL_SPLIT_ID_FILE = SPLIT_DIR + '/val_imgs.txt'
TEST_SPLIT_ID_FILE = SPLIT_DIR + '/test_imgs.txt'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
train_img_ids = np.loadtxt(TRAIN_SPLIT_ID_FILE, dtype=np.int32)
val_img_ids = np.loadtxt(VAL_SPLIT_ID_FILE, dtype=np.int32)
test_img_ids = np.loadtxt(TEST_SPLIT_ID_FILE, dtype=np.int32)

########## HYPERPARAMETERS ##########
N_EPOCHS = args.n_epochs
BACKBONE = args.backbone
TRAINABLE_CONVNET = bool(args.trainable_convnet)
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
MAX_BG_BOXES = args.max_bg_boxes if args.max_bg_boxes > 0 else -1
WEIGHT_DECAY = args.weight_decay
ROI_POOL_OUTPUT_SIZE = (args.roi, args.roi)
DROP_PROB = args.drop_prob
POS_FEAT = bool(args.pos_feat)

params = '%s %s lr-%.0e batch-%d wd-%.0e roi-%d dp-%.2f pf-%d mbb-%d' % (args.split, BACKBONE, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, ROI_POOL_OUTPUT_SIZE[0], DROP_PROB, POS_FEAT, MAX_BG_BOXES)
log_file = '%s/%s logs.txt' % (OUTPUT_DIR, params)
model_save_file = '%s/%s saved_model.pth' % (OUTPUT_DIR, params)

print('logs will be saved in \"%s\"' % (log_file))
print_and_log('Backbone Convnet: %s' % (BACKBONE), log_file, 'w')
print_and_log('Trainable Convnet: %s' % (TRAINABLE_CONVNET), log_file)
print_and_log('Learning Rate: %.0e' % (LEARNING_RATE), log_file)
print_and_log('Batch Size: %d' % (BATCH_SIZE), log_file)
print_and_log('Max BG Boxes: %d' % (MAX_BG_BOXES), log_file)
print_and_log('Weight Decay: %.0e' % (WEIGHT_DECAY), log_file)
print_and_log('RoI Pool Output Size: (%d, %d)' % ROI_POOL_OUTPUT_SIZE, log_file)
print_and_log('Dropout Probability: %.2f' % (DROP_PROB), log_file)
print_and_log('Position Features: %s\n' % (POS_FEAT), log_file)

########## DATA LOADERS ##########
train_loader, val_loader, test_loader = load_data(DATA_DIR, train_img_ids, val_img_ids, test_img_ids, BATCH_SIZE, NUM_WORKERS, MAX_BG_BOXES)

########## CREATE MODEL & LOSS FN ##########
model = WebObjExtractionNet(ROI_POOL_OUTPUT_SIZE, IMG_HEIGHT, N_CLASSES, BACKBONE, TRAINABLE_CONVNET, DROP_PROB, POS_FEAT, CLASS_NAMES).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

########## TRAIN MODEL ##########
train_model(model, train_loader, optimizer, criterion, N_EPOCHS, device, val_loader, EVAL_INTERVAL, log_file, 'ckpt_%d.pth' % args.device)

########## EVALUATE TEST PERFORMANCE ##########
evaluate_model(model, test_loader, criterion, device, 'TEST', log_file)

########## SAVE MODEL ##########
torch.save(model.state_dict(), model_save_file)
print_and_log('Model can be restored from \"%s\"' % (model_save_file), log_file)
