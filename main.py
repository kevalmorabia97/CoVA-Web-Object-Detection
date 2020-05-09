import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_data
from models import WebObjExtractionNet
from train import train_model, evaluate_model
from utils import print_and_log, set_all_seeds


########## CMDLINE ARGS ##########
parser = argparse.ArgumentParser('Train Model')
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('-e', '--n_epochs', type=int, default=100)
parser.add_argument('-bb', '--backbone', type=str, default='resnet', choices=['alexnet', 'resnet'])
parser.add_argument('--freeze_convnet', dest='freeze_convnet', action='store_true')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005)
parser.add_argument('-bs', '--batch_size', type=int, default=5)
parser.add_argument('--no_context', dest='use_context', action='store_false')
parser.add_argument('-cs', '--context_size', type=int, default=12)
parser.add_argument('--no_attention', dest='attention', action='store_false')
parser.add_argument('-atth', '--attention_heads', type=int, default=1)
parser.add_argument('-hd', '--hidden_dim', type=int, default=384)
parser.add_argument('-r', '--roi', type=int, default=3)
parser.add_argument('--no_bbox_feat', dest='bbox_feat', action='store_false')
parser.add_argument('-bbhd', '--bbox_hidden_dim', type=int, default=32)
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3)
parser.add_argument('-dp', '--drop_prob', type=float, default=0.2)
parser.add_argument('-sf', '--sampling_fraction', type=float, default=0.8)
parser.add_argument('-nw', '--num_workers', type=int, default=5)
parser.add_argument('-cvf', '--cv_fold', type=int, required=True, choices=[1,2,3,4,5])
args = parser.parse_args()

device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
set_all_seeds(123)

########## PARAMETERS ##########
N_CLASSES = 4
CLASS_NAMES = ['BG', 'Price', 'Title', 'Image']
IMG_HEIGHT = 1280 # Image assumed to have same height and width
EVAL_INTERVAL = 2 # Number of Epochs after which model is evaluated
NUM_WORKERS = args.num_workers # multithreaded data loading

DATA_DIR = '../data/' # Contains imgs/*.png and bboxes/*.pkl files
CV_FOLD = args.cv_fold
SPLIT_DIR = 'splits'
FOLD_DIR = '%s/Fold-%d' % (SPLIT_DIR, CV_FOLD)
OUTPUT_DIR = 'results_5-Fold_CV' # results are saved here!
# NOTE: if same hyperparameter configuration is run again, previous log file and saved model will be overwritten

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
ATTENTION_HEADS = args.attention_heads if USE_CONTEXT and USE_ATTENTION else 0
HIDDEN_DIM = args.hidden_dim if USE_CONTEXT and USE_ATTENTION else 0
ROI_OUTPUT = (args.roi, args.roi)
USE_BBOX_FEAT = args.bbox_feat
BBOX_HIDDEN_DIM = args.bbox_hidden_dim if USE_BBOX_FEAT else 0
WEIGHT_DECAY = args.weight_decay
DROP_PROB = args.drop_prob
SAMPLING_FRACTION = args.sampling_fraction if (args.sampling_fraction >= 0 and args.sampling_fraction <= 1) else 1

params = '%s lr-%.0e batch-%d c-%d cs-%d att-%d atth-%d hd-%d roi-%d bbf-%d bbhd-%d wd-%.0e dp-%.1f sf-%.1f' % (BACKBONE, LEARNING_RATE, BATCH_SIZE,
    USE_CONTEXT, CONTEXT_SIZE, USE_ATTENTION, ATTENTION_HEADS, HIDDEN_DIM, ROI_OUTPUT[0], USE_BBOX_FEAT, BBOX_HIDDEN_DIM, WEIGHT_DECAY, DROP_PROB,
    SAMPLING_FRACTION)
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
                                                  BATCH_SIZE, NUM_WORKERS, SAMPLING_FRACTION)

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
print_and_log('Attention Heads: %d' % (ATTENTION_HEADS), log_file)
print_and_log('Hidden Dim: %d' % (HIDDEN_DIM), log_file)
print_and_log('RoI Pool Output Size: (%d, %d)' % ROI_OUTPUT, log_file)
print_and_log('Use BBox Features: %s' % (USE_BBOX_FEAT), log_file)
print_and_log('BBox Hidden Dim: %d' % (BBOX_HIDDEN_DIM), log_file)
print_and_log('Weight Decay: %.0e' % (WEIGHT_DECAY), log_file)
print_and_log('Dropout Probability: %.2f' % (DROP_PROB), log_file)
print_and_log('Sampling Fraction: %.2f\n' % (SAMPLING_FRACTION), log_file)

########## TRAIN MODEL ##########
model = WebObjExtractionNet(ROI_OUTPUT, IMG_HEIGHT, N_CLASSES, BACKBONE, USE_CONTEXT, USE_ATTENTION, ATTENTION_HEADS, HIDDEN_DIM, USE_BBOX_FEAT,
                            BBOX_HIDDEN_DIM, TRAINABLE_CONVNET, DROP_PROB, CLASS_NAMES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1) # No LR Scheduling
criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

val_acc = train_model(model, train_loader, optimizer, scheduler, criterion, N_EPOCHS, device, val_loader, EVAL_INTERVAL, log_file, model_save_file)

print('Evaluating on test data...')
img_acc, class_acc = evaluate_model(model, test_loader, device, 1, 'TEST', log_file)
_, class_acc_top2 = evaluate_model(model, test_loader, device, 2, 'TEST', log_file)

np.savetxt(test_acc_imgwise_file, img_acc, '%d,%.2f,%.2f,%.2f', ',', header='img_id,price_acc,title_acc,image_acc', comments='')

with open (test_acc_domainwise_file, 'w') as f:
    f.write('Domain,N_examples,%s,%s,%s\n' % (CLASS_NAMES[1], CLASS_NAMES[2], CLASS_NAMES[3]))
    for domain in test_domains:
        domain_imgs = webpage_info[np.isin(webpage_info[:,1], domain), 0].astype(np.int32)
        domain_class_acc = img_acc[np.isin(img_acc[:,0], domain_imgs), 1:].mean(0)*100
        f.write('%s,%d,%.2f,%.2f,%.2f\n' % (domain, len(domain_imgs), domain_class_acc[0], domain_class_acc[1], domain_class_acc[2]))
macro_acc_test = np.loadtxt(test_acc_domainwise_file, delimiter=',', skiprows=1, dtype=str)[:,2:].astype(np.float32).mean(0)
for c in range(1, N_CLASSES):
    print_and_log('%s Macro Acc: %.2f%%' % (CLASS_NAMES[c], macro_acc_test[c-1]), log_file)

with open(fold_wise_acc_file, 'a') as f:
    f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (CV_FOLD, val_acc, class_acc[0], class_acc_top2[0], macro_acc_test[0],
        class_acc[1], class_acc_top2[1], macro_acc_test[1], class_acc[2], class_acc_top2[2], macro_acc_test[2]))
