import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import custom_collate_fn, WebDataset
from models import WebObjExtractionNet
from train import evaluate_model

DEVICE_NO = 0
device = torch.device('cuda:%d' % DEVICE_NO if torch.cuda.is_available() else 'cpu')

DATA_DIR = '/shared/data_product_info/v2_8.3k/'
OUTPUT_DIR = 'results/' # location where saved model .pth file is located
N_CLASSES = 4
CLASS_NAMES = ['BG', 'Price', 'Title', 'Image']
IMG_HEIGHT = 1280

# Load the model for which evaluation has to be done
BACKBONE = 'alexnet'
TRAINABLE_CONVNET = True
LEARNING_RATE = 5e-4
BATCH_SIZE = 50
MAX_BG_BOXES = 100
WEIGHT_DECAY = 1e-1
ROI_POOL_OUTPUT_SIZE = (1,1)
DROP_PROB = 0.2
POS_FEAT = True

params = 'domain_wise %s lr-%.0e batch-%d wd-%.0e roi-%d dp-%.2f pf-%d mbb-%d' % (BACKBONE, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, ROI_POOL_OUTPUT_SIZE[0], DROP_PROB, POS_FEAT, MAX_BG_BOXES)
model_save_file = '%s/%s saved_model.pth' % (OUTPUT_DIR, params)
output = '%s/%s per_domain_acc.csv' % (OUTPUT_DIR, params)

model = WebObjExtractionNet(ROI_POOL_OUTPUT_SIZE, IMG_HEIGHT, N_CLASSES, BACKBONE, TRAINABLE_CONVNET, DROP_PROB, POS_FEAT, CLASS_NAMES).to(device)
model.load_state_dict(torch.load(model_save_file, map_location=device))

criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

test_domains = [f.name for f in os.scandir('splits/domain_wise/per_domain_test/') if '.txt' in f.name]
print('Evaluating per domain accuracy for %d domains' % len(test_domains))

with open (output, 'w') as f:
    f.write('domain,n_examples,acc,%s,%s,%s\n' % (CLASS_NAMES[1].lower(), CLASS_NAMES[2].lower(), CLASS_NAMES[3].lower()))

for domain in test_domains:
    print('\n---> Domain:', domain)
    test_dataset = WebDataset(DATA_DIR, np.loadtxt('splits/domain_wise/per_domain_test/%s' % domain, np.int32).reshape(-1), max_bg_boxes=-1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=custom_collate_fn, drop_last=False)

    per_class_accuracy = evaluate_model(model, test_loader, criterion, device, 'TEST')
    avg = per_class_accuracy[1:].mean()
    p = per_class_accuracy[1]
    t = per_class_accuracy[2]
    i = per_class_accuracy[3]

    with open (output, 'a') as f:
        f.write('%s,%d,%.2f,%.2f,%.2f,%.2f\n' % (domain, len(test_dataset), 100*avg, 100*p, 100*t, 100*i))
    
macro_acc = np.loadtxt(output, delimiter=',', skiprows=1, dtype=str)[:,2:].astype(np.float32).mean(0)
for i in range(1, len(CLASS_NAMES)):
    print('%s Macro Acc: %.4f' % (CLASS_NAMES[i], macro_acc[i]))