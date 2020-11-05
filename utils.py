import argparse
import numpy as np
import pickle
from PIL import Image
import random
import torch


def cmdline_args_parser():
    """
    Commandline argument parser
    """
    parser = argparse.ArgumentParser('Cmdline Arguments')
    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('-e', '--n_epochs', type=int, default=100)
    parser.add_argument('-bb', '--backbone', type=str, default='resnet', choices=['alexnet', 'resnet'])
    parser.add_argument('--freeze_convnet', dest='freeze_convnet', action='store_true')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005)
    parser.add_argument('-bs', '--batch_size', type=int, default=5)
    parser.add_argument('--no_context', dest='use_context', action='store_false')
    parser.add_argument('-cs', '--context_size', type=int, default=12)
    parser.add_argument('--no_attention', dest='attention', action='store_false')
    parser.add_argument('-hd', '--hidden_dim', type=int, default=384)
    parser.add_argument('-r', '--roi', type=int, default=3)
    parser.add_argument('--no_bbox_feat', dest='bbox_feat', action='store_false')
    parser.add_argument('-bbhd', '--bbox_hidden_dim', type=int, default=32)
    parser.add_argument('--use_additional_feat', dest='additional_feat', action='store_true')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3)
    parser.add_argument('-dp', '--drop_prob', type=float, default=0.2)
    parser.add_argument('-sf', '--sampling_fraction', type=float, default=0.8)
    parser.add_argument('-nw', '--num_workers', type=int, default=5)
    parser.add_argument('-cvf', '--cv_fold', type=int, required=True, choices=[-1,1,2,3,4,5]) # cvf=-1 means fold_dir is set to split_dir
    
    return parser


def compute_image_data_statistics(data_loader):
    """
    Return the channel wise mean and std deviation for images loaded by `data_loader` (loads WebDataset defined in `datasets.py`)
    Should be computer on train+val data
    """
    mean = 0.
    std = 0.
    n_samples = 0.

    for images, bboxes, labels in data_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_samples

    mean /= n_samples
    std /= n_samples

    return mean, std


def count_parameters(model):
    """
    Return the number of trainable parameters in `model`
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pkl_load(file_path):
    """
    Load a pickle file at filt_path
    """
    return pickle.load(open(file_path, 'rb'))


def print_and_log(msg, log_file, write_mode='a'):
    """
    print `msg` (string) on stdout and also append ('a') or write ('w') (default 'a') it to `log_file`
    """
    print(msg)
    with open(log_file, write_mode) as f:
        f.write(msg + '\n')


def set_all_seeds(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def visualize_bbox(img_path, attn_wt_file, img_save_dir):
    """
    Plot img and show all context bboxes on the img with attention scores
    Target BBox is bold black, context bbox is either green (score >= 0.2) or red (score < 0.2)
    attn_wt_file is a csv file containing 3 rows, 5 + 10*context_size cols
    Each row contains plot data for a target class (Price, Title, Image)
    Cols: 4 bbox coords, 1 label, 2*context_size*4 context bbox coords, 2*context_size attnetion values that sum to 1

    Save 3 files corresponding to 3 classes in img_save_dir (must exist)
    """
    import matplotlib.pyplot as plt
    
    class_names = {0:'BG', 1:'Price', 2:'Title', 3:'Image'}

    img = Image.open(img_path).convert('RGB')
    plt_data = np.loadtxt(attn_wt_file, delimiter=',')
    context_size = int((plt_data.shape[1] - 5) / 10)

    plt_data[:,-2*context_size:] /= plt_data[:,-2*context_size:].max()

    plt.rcParams.update({'font.size': 6})
    for row in plt_data:
        plt.imshow(img)
        plt.title('Attention Visualization for class: ' + class_names[int(row[4])])
        ax = plt.gca()
        ax.add_patch(plt.Rectangle((row[0], row[1]), row[2], row[3], fill=False, edgecolor='black', linewidth=1.5))
        for c in range(1, 2*context_size+1):
            if row[4*c+1] == 0 and row[4*c+2] == 0 and row[4*c+3] == 0 and row[4*c+4] == 0:
                continue
            ax.add_patch(plt.Rectangle((row[4*c+1], row[4*c+2]), row[4*c+3], row[4*c+4], fill=True, facecolor='red', alpha=0.75*row[4*(2*context_size+1) + c]))
            ax.add_patch(plt.Rectangle((row[4*c+1], row[4*c+2]), row[4*c+3], row[4*c+4], fill=False, edgecolor='red', linewidth=0.75))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('%s/%s_attn_%s.png' % (img_save_dir, img_path.rsplit('/',1)[-1][:-4], class_names[int(row[4])]), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()