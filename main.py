import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import custom_collate_fn, load_data, WebDataset
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
parser.add_argument('-cs', '--context_size', type=int, default=6)
parser.add_argument('-att', '--attention', type=int, default=1, choices=[0,1])
parser.add_argument('-hd', '--hidden_dim', type=int, default=300)
parser.add_argument('-r', '--roi', type=int, default=1)
parser.add_argument('-bbf', '--bbox_feat', type=int, default=1, choices=[0,1])
parser.add_argument('-wd', '--weight_decay', type=float, default=0)
parser.add_argument('-dp', '--drop_prob', type=float, default=0.5)
parser.add_argument('-mbb', '--max_bg_boxes', type=int, default=-1)
parser.add_argument('-nw', '--num_workers', type=int, default=8)
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

DATA_DIR = '/shared/data_product_info/v2_8.3k/' # Contains .png and .pkl files for train and test data
OUTPUT_DIR = 'results_attn' # logs are saved here!
# NOTE: if same hyperparameter configuration is run again, previous log file and saved model will be overwritten

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SPLIT_DIR = 'splits'
train_img_ids = np.loadtxt('%s/train_imgs.txt' % SPLIT_DIR, dtype=np.int32)
val_img_ids = np.loadtxt('%s/val_imgs.txt' % SPLIT_DIR, dtype=np.int32)
test_img_ids = np.loadtxt('%s/test_imgs.txt' % SPLIT_DIR, dtype=np.int32)

test_domains = np.loadtxt('%s/test_domains.txt' % SPLIT_DIR, dtype=str) # for calculating macro accuracy

########## HYPERPARAMETERS ##########
N_EPOCHS = args.n_epochs
BACKBONE = args.backbone
TRAINABLE_CONVNET = bool(args.trainable_convnet)
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
CONTEXT_SIZE = args.context_size
USE_ATTENTION = bool(args.attention)
HIDDEN_DIM = args.hidden_dim
ROI_POOL_OUTPUT_SIZE = (args.roi, args.roi)
USE_BBOX_FEAT = bool(args.bbox_feat)
WEIGHT_DECAY = args.weight_decay
DROP_PROB = args.drop_prob
MAX_BG_BOXES = args.max_bg_boxes if args.max_bg_boxes > 0 else -1

params = '%s lr-%.0e batch-%d cs-%d att-%d hd-%d roi-%d bbf-%d wd-%.0e dp-%.2f mbb-%d' % (BACKBONE, LEARNING_RATE, BATCH_SIZE, CONTEXT_SIZE, USE_ATTENTION,
    HIDDEN_DIM, ROI_POOL_OUTPUT_SIZE[0], USE_BBOX_FEAT, WEIGHT_DECAY, DROP_PROB, MAX_BG_BOXES)
log_file = '%s/%s logs.txt' % (OUTPUT_DIR, params)
test_acc_domainwise_file = '%s/%s test_acc_domainwise.csv' % (OUTPUT_DIR, params)
model_save_file = '%s/%s saved_model.pth' % (OUTPUT_DIR, params)

print('logs will be saved in \"%s\"' % (log_file))
print_and_log('Backbone Convnet: %s' % (BACKBONE), log_file, 'w')
print_and_log('Trainable Convnet: %s' % (TRAINABLE_CONVNET), log_file)
print_and_log('Learning Rate: %.0e' % (LEARNING_RATE), log_file)
print_and_log('Batch Size: %d' % (BATCH_SIZE), log_file)
print_and_log('Context Size: %d' % (CONTEXT_SIZE), log_file)
print_and_log('Attention: %s' % (USE_ATTENTION), log_file)
print_and_log('Hidden Dim: %d' % (HIDDEN_DIM), log_file)
print_and_log('RoI Pool Output Size: (%d, %d)' % ROI_POOL_OUTPUT_SIZE, log_file)
print_and_log('BBox Features: %s' % (USE_BBOX_FEAT), log_file)
print_and_log('Weight Decay: %.0e' % (WEIGHT_DECAY), log_file)
print_and_log('Dropout Probability: %.2f' % (DROP_PROB), log_file)
print_and_log('Max BG Boxes: %d\n' % (MAX_BG_BOXES), log_file)

########## DATA LOADERS ##########
train_loader, val_loader, test_loader = load_data(DATA_DIR, train_img_ids, val_img_ids, test_img_ids, CONTEXT_SIZE, BATCH_SIZE, NUM_WORKERS, MAX_BG_BOXES)

########## CREATE MODEL & LOSS FN ##########
model = WebObjExtractionNet(ROI_POOL_OUTPUT_SIZE, IMG_HEIGHT, N_CLASSES, BACKBONE, USE_ATTENTION, HIDDEN_DIM, TRAINABLE_CONVNET, DROP_PROB,
                            USE_BBOX_FEAT, CLASS_NAMES).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

########## TRAIN MODEL ##########
train_model(model, train_loader, optimizer, criterion, N_EPOCHS, device, val_loader, EVAL_INTERVAL, log_file, 'ckpt_%d.pth' % args.device)

########## EVALUATE TEST PERFORMANCE ##########
print('Evaluating test data class wise accuracies...')
evaluate_model(model, test_loader, criterion, device, 'TEST', log_file)

with open (test_acc_domainwise_file, 'w') as f:
    f.write('Domain,N_examples,%s,%s,%s\n' % (CLASS_NAMES[1], CLASS_NAMES[2], CLASS_NAMES[3]))

print('Evaluating per domain accuracy for %d test domains...' % len(test_domains))
for domain in test_domains:
    print('\n---> Domain:', domain)
    test_dataset = WebDataset(DATA_DIR, np.loadtxt('%s/domain_wise_imgs/%s.txt' % (SPLIT_DIR, domain), np.int32).reshape(-1), CONTEXT_SIZE, max_bg_boxes=-1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_collate_fn, drop_last=False)

    per_class_acc = evaluate_model(model, test_loader, criterion, device, 'TEST')

    with open (test_acc_domainwise_file, 'a') as f:
        f.write('%s,%d,%.2f,%.2f,%.2f\n' % (domain, len(test_dataset), 100*per_class_acc[1], 100*per_class_acc[2], 100*per_class_acc[3]))

macro_acc_test = np.loadtxt(test_acc_domainwise_file, delimiter=',', skiprows=1, dtype=str)[:,2:].astype(np.float32).mean(0)
for i in range(1, len(CLASS_NAMES)):
    print_and_log('%s Macro Acc: %.2f%%' % (CLASS_NAMES[i], macro_acc_test[i-1]), log_file)

########## SAVE MODEL ##########
torch.save(model.state_dict(), model_save_file)
print_and_log('Model can be restored from \"%s\"' % (model_save_file), log_file)
