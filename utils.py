import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image


def compute_image_data_statistics(data_loader):
    """
    Return the channel wise mean and std deviation for images loaded by `data_loader` (loads WebDataset defined in `datasets.py`)
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


def print_confusion_matrix(c, class_names=None):
    """
    c: np.array of shape [n_classes, n_classes] where
        each row represents True labels
        each col represents Pred labels
    
    class_names: list of n_classes items each containing name of classes in order
                    if None (default), class names will be set to 0, 1, ..., n_classes-1
    """
    c = c.astype(str)
    n_classes = c.shape[0]
    if class_names is None:
        class_names = np.arange(n_classes).astype(str)
    
    print( 'True \\ Pred\t%s' % ('\t'.join(class_names)) )
    for i in range(n_classes):
        print( '%s\t\t%s' % (class_names[i], '\t'.join(c[i])) )


def visualize_bbox(img_path, attn_wt_file, img_save_dir):
    """
    Plot img and show all context bboxes on the img with attention scores
    Target BBox is bold black, context bbox is either green (score >= 0.2) or red (score < 0.2)
    attn_wt_file is a csv file containing 3 rows, 5 + 10*context_size cols
    Each row contains plot data for a target class (Price, Title, Image)
    Cols: 4 bbox coords, 1 label, 2*context_size*4 context bbox coords, 2*context_size attnetion values that sum to 1

    Save 3 files corresponding to 3 classes in img_save_dir (must exist)
    """
    class_names = {0:'BG', 1:'Price', 2:'Title', 3:'Image'}

    img = Image.open(img_path).convert('RGB')
    plt_data = np.loadtxt(attn_wt_file, delimiter=',')
    context_size = int((plt_data.shape[1] - 5) / 10)

    plt.rcParams.update({'font.size': 6})
    for row in plt_data:
        plt.imshow(img)
        plt.title('Attention Visualization for class: ' + class_names[int(row[4])])
        ax = plt.gca()
        ax.add_patch(plt.Rectangle((row[0], row[1]), row[2], row[3], fill=False, edgecolor='black', linewidth=2))
        for c in range(1, 2*context_size+1):
            if row[4*c+1] == 0 and row[4*c+2] == 0 and row[4*c+3] == 0 and row[4*c+4] == 0:
                continue
            if row[4*(2*context_size+1) + c] >= 0.2:
                ax.text(row[4*c+1], row[4*c+2], '%.1f' % (100*row[4*(2*context_size+1) + c]))
                color = 'green'
            else:
                color = 'red'
            ax.add_patch(plt.Rectangle((row[4*c+1], row[4*c+2]), row[4*c+3], row[4*c+4], fill=False, edgecolor=color, linewidth=1))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('%s/%s_attn_%s.png' % (img_save_dir, img_path.rsplit('/',1)[-1][:-4], class_names[int(row[4])]), dpi=300, bbox_inches = 'tight', pad_inches = 0)
        plt.close()
