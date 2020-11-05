import numpy as np
import os
import torch
import torch.nn as nn

from constants import Constants
from datasets import load_data
from evaluate import evaluate
from models import WebObjExtractionNet
from train import train_model
from utils import cmdline_args_parser, print_and_log, set_all_seeds


parser = cmdline_args_parser()
args = parser.parse_args()

device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
set_all_seeds(Constants.SEED)

N_CLASSES = Constants.N_CLASSES
CLASS_NAMES = Constants.CLASS_NAMES
IMG_HEIGHT = Constants.IMG_HEIGHT
DATA_DIR = Constants.DATA_DIR
SPLIT_DIR = Constants.SPLIT_DIR
OUTPUT_DIR = Constants.OUTPUT_DIR
# NOTE: if same hyperparameter configuration is run again, previous log file and saved model will be overwritten

EVAL_INTERVAL = 1 # Number of Epochs after which model is evaluated while training
NUM_WORKERS = args.num_workers # multithreaded data loading

CV_FOLD = args.cv_fold
FOLD_DIR = '%s/Fold-%d' % (SPLIT_DIR, CV_FOLD)

# for calculating macro accuracy
webpage_info = np.loadtxt('%s/webpage_info.csv' % SPLIT_DIR, str, delimiter=',', skiprows=1) # (img_id, domain) values

########## HYPERPARAMETERS ##########
N_EPOCHS = args.n_epochs
BACKBONE = args.backbone
TRAINABLE_CONVNET = not args.freeze_convnet
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
USE_CONTEXT = args.use_context
CONTEXT_SIZE = args.context_size if USE_CONTEXT else 0
USE_ATTENTION = args.attention if USE_CONTEXT else False
HIDDEN_DIM = args.hidden_dim if USE_CONTEXT and USE_ATTENTION else 0
ROI_OUTPUT = (args.roi, args.roi)
USE_BBOX_FEAT = args.bbox_feat
BBOX_HIDDEN_DIM = args.bbox_hidden_dim if USE_BBOX_FEAT else 0
USE_ADDITIONAL_FEAT = args.additional_feat
WEIGHT_DECAY = args.weight_decay
DROP_PROB = args.drop_prob
SAMPLING_FRACTION = args.sampling_fraction if (args.sampling_fraction >= 0 and args.sampling_fraction <= 1) else 1

params = '%s lr-%.0e batch-%d c-%d cs-%d att-%d hd-%d roi-%d bbf-%d bbhd-%d af-%d wd-%.0e dp-%.1f sf-%.1f' % (BACKBONE, LEARNING_RATE,
    BATCH_SIZE, USE_CONTEXT, CONTEXT_SIZE, USE_ATTENTION, HIDDEN_DIM, ROI_OUTPUT[0], USE_BBOX_FEAT, BBOX_HIDDEN_DIM,
    USE_ADDITIONAL_FEAT, WEIGHT_DECAY, DROP_PROB, SAMPLING_FRACTION)
results_dir = '%s/%s' % (OUTPUT_DIR, params)
fold_wise_acc_file = '%s/fold_wise_acc.csv' % results_dir

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print('\n%s Training on Fold-%s %s' % ('*'*20, CV_FOLD, '*'*20))
########## DATA LOADERS ##########
train_img_ids = np.loadtxt('%s/train_imgs.txt' % FOLD_DIR, np.int32)
val_img_ids = np.loadtxt('%s/val_imgs.txt' % FOLD_DIR, np.int32)
test_img_ids = np.loadtxt('%s/test_imgs.txt' % FOLD_DIR, np.int32)
test_domains = np.loadtxt('%s/test_domains.txt' % FOLD_DIR, str)

train_loader, val_loader, test_loader = load_data(DATA_DIR, train_img_ids, val_img_ids, test_img_ids, USE_CONTEXT, CONTEXT_SIZE,
                                                  BATCH_SIZE, USE_ADDITIONAL_FEAT, SAMPLING_FRACTION, NUM_WORKERS)
n_additional_features = train_loader.dataset.n_additional_features

log_file = '%s/Fold-%s logs.txt' % (results_dir, CV_FOLD)
test_acc_imgwise_file = '%s/Fold-%s test_acc_imgwise.csv' % (results_dir, CV_FOLD)
test_acc_domainwise_file = '%s/Fold-%s test_acc_domainwise.csv' % (results_dir, CV_FOLD)
model_save_file = '%s/Fold-%s saved_model.pth' % (results_dir, CV_FOLD)

print('logs will be saved in \"%s\"' % (log_file))
print_and_log('Backbone Convnet: %s' % (BACKBONE), log_file, 'w')
print_and_log('Trainable Convnet: %s' % (TRAINABLE_CONVNET), log_file)
print_and_log('Learning Rate: %.0e' % (LEARNING_RATE), log_file)
print_and_log('Batch Size: %d' % (BATCH_SIZE), log_file)
print_and_log('Use Context: %s' % (USE_CONTEXT), log_file)
print_and_log('Context Size: %d' % (CONTEXT_SIZE), log_file)
print_and_log('Use Attention: %s' % (USE_ATTENTION), log_file)
print_and_log('Hidden Dim: %d' % (HIDDEN_DIM), log_file)
print_and_log('RoI Pool Output Size: (%d, %d)' % ROI_OUTPUT, log_file)
print_and_log('Use BBox Features: %s' % (USE_BBOX_FEAT), log_file)
print_and_log('BBox Hidden Dim: %d' % (BBOX_HIDDEN_DIM), log_file)
print_and_log('Use Additional Features: %s' % (USE_ADDITIONAL_FEAT), log_file)
print_and_log('Weight Decay: %.0e' % (WEIGHT_DECAY), log_file)
print_and_log('Dropout Probability: %.2f' % (DROP_PROB), log_file)
print_and_log('Sampling Fraction: %.2f\n' % (SAMPLING_FRACTION), log_file)

########## TRAIN MODEL ##########
model = WebObjExtractionNet(ROI_OUTPUT, IMG_HEIGHT, N_CLASSES, BACKBONE, USE_CONTEXT, USE_ATTENTION, HIDDEN_DIM, USE_BBOX_FEAT,
                            BBOX_HIDDEN_DIM, n_additional_features, TRAINABLE_CONVNET, DROP_PROB, CLASS_NAMES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1) # No LR Scheduling
criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

val_acc = train_model(model, train_loader, optimizer, scheduler, criterion, N_EPOCHS, device, val_loader, EVAL_INTERVAL,
                      log_file, model_save_file)

class_acc_test, macro_acc_test = evaluate(model, test_loader, test_domains, webpage_info, device, log_file, 
                                          test_acc_imgwise_file, test_acc_domainwise_file)

with open(fold_wise_acc_file, 'a') as f:
    if os.stat(fold_wise_acc_file).st_size == 0: # add header if file is empty
        f.write('Fold,val_avg,price_acc,price_macro_acc,title_acc,title_macro_acc,image_acc,image_macro_acc\n')

    f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (CV_FOLD, val_acc, class_acc_test[0], macro_acc_test[0],
        class_acc_test[1], macro_acc_test[1], class_acc_test[2], macro_acc_test[2]))
