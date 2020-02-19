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
parser.add_argument('-bb', '--backbone', type=str, default='resnet', choices=['alexnet', 'resnet'])
parser.add_argument('-tc', '--trainable_convnet', type=int, default=1, choices=[0,1])
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005)
parser.add_argument('-bs', '--batch_size', type=int, default=25)
parser.add_argument('-c', '--context', type=int, default=1, choices=[0,1])
parser.add_argument('-cs', '--context_size', type=int, default=6)
parser.add_argument('-att', '--attention', type=int, default=1, choices=[0,1])
parser.add_argument('-hd', '--hidden_dim', type=int, default=200)
parser.add_argument('-r', '--roi', type=int, default=3)
parser.add_argument('-bbf', '--bbox_feat', type=int, default=1, choices=[0,1])
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3)
parser.add_argument('-dp', '--drop_prob', type=float, default=0.2)
parser.add_argument('-mbb', '--max_bg_boxes', type=int, default=100)
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
EVAL_INTERVAL = 2 # Number of Epochs after which model is evaluated
NUM_WORKERS = args.num_workers # multithreaded data loading

DATA_DIR = '../data/v3/' # Contains imgs/*.png and bboxes/*.pkl files
SPLIT_DIR = 'splits'
CV_FOLDS = ['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'] # these dirs are inside `SPLIT_DIR`
OUTPUT_DIR = 'results_5-Fold_CV' # results are saved here!
# NOTE: if same hyperparameter configuration is run again, previous log file and saved model will be overwritten

# for calculating macro accuracy
webpage_info = np.loadtxt('%s/webpage_info.csv' % SPLIT_DIR, str, delimiter=',', skiprows=1) # (img_id, domain) values

########## HYPERPARAMETERS ##########
N_EPOCHS = args.n_epochs
BACKBONE = args.backbone
TRAINABLE_CONVNET = bool(args.trainable_convnet)
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
USE_CONTEXT = bool(args.context)
CONTEXT_SIZE = args.context_size if USE_CONTEXT else 0
USE_ATTENTION = bool(args.attention) if USE_CONTEXT else False
HIDDEN_DIM = args.hidden_dim if USE_CONTEXT and USE_ATTENTION else 0
ROI_OUTPUT = (args.roi, args.roi)
USE_BBOX_FEAT = bool(args.bbox_feat)
WEIGHT_DECAY = args.weight_decay
DROP_PROB = args.drop_prob
MAX_BG_BOXES = args.max_bg_boxes if args.max_bg_boxes > 0 else -1

params = '%s lr-%.0e batch-%d c-%d cs-%d att-%d hd-%d roi-%d bbf-%d wd-%.0e dp-%.2f mbb-%d' % (BACKBONE, LEARNING_RATE, BATCH_SIZE,
    USE_CONTEXT, CONTEXT_SIZE, USE_ATTENTION, HIDDEN_DIM, ROI_OUTPUT[0], USE_BBOX_FEAT, WEIGHT_DECAY, DROP_PROB, MAX_BG_BOXES)
results_dir = '%s/%s' % (OUTPUT_DIR, params)
fold_wise_acc_file = '%s/fold_wise_acc.csv' % results_dir

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

for fold in CV_FOLDS:
    print('\n%s Training on %s %s' % ('*'*20, fold, '*'*20))
    ########## DATA LOADERS ##########
    train_img_ids = np.loadtxt('%s/%s/train_imgs.txt' % (SPLIT_DIR, fold), np.int32)
    val_img_ids = np.loadtxt('%s/%s/val_imgs.txt' % (SPLIT_DIR, fold), np.int32)
    test_img_ids = np.loadtxt('%s/%s/test_imgs.txt' % (SPLIT_DIR, fold), np.int32)
    test_domains = np.loadtxt('%s/%s/test_domains.txt' % (SPLIT_DIR, fold), str)

    train_loader, val_loader, test_loader = load_data(DATA_DIR, train_img_ids, val_img_ids, test_img_ids, USE_CONTEXT, CONTEXT_SIZE,
                                                      BATCH_SIZE, NUM_WORKERS, MAX_BG_BOXES)

    log_file = '%s/%s logs.txt' % (results_dir, fold)
    test_acc_imgwise_file = '%s/%s test_acc_imgwise.csv' % (results_dir, fold)
    test_acc_domainwise_file = '%s/%s test_acc_domainwise.csv' % (results_dir, fold)
    model_save_file = '%s/%s saved_model.pth' % (results_dir, fold)

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
    print_and_log('BBox Features: %s' % (USE_BBOX_FEAT), log_file)
    print_and_log('Weight Decay: %.0e' % (WEIGHT_DECAY), log_file)
    print_and_log('Dropout Probability: %.2f' % (DROP_PROB), log_file)
    print_and_log('Max BG Boxes: %d\n' % (MAX_BG_BOXES), log_file)

    ########## TRAIN MODEL ##########
    model = WebObjExtractionNet(ROI_OUTPUT, IMG_HEIGHT, N_CLASSES, BACKBONE, USE_CONTEXT, USE_ATTENTION, HIDDEN_DIM, TRAINABLE_CONVNET,
                                DROP_PROB, USE_BBOX_FEAT, CLASS_NAMES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    val_acc = train_model(model, train_loader, optimizer, scheduler, criterion, N_EPOCHS, device, val_loader, EVAL_INTERVAL, log_file, 'ckpt_%d.pth' % args.device)
    torch.save(model.state_dict(), model_save_file)

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
        f.write('%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (fold, val_acc, class_acc[0], class_acc_top2[0], macro_acc_test[0],
            class_acc[1], class_acc_top2[1], macro_acc_test[1], class_acc[2], class_acc_top2[2], macro_acc_test[2]))
