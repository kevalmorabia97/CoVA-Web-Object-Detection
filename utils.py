import matplotlib.pyplot as plt
import os
import pickle


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


def create_dir(dir_path):
    """
    Create directory structure given as `dir_path`.
    If path exists ask whether to overwrite it or make a new directory by appending time to it.

    return: path to directory created
    """
    if os.path.exists(dir_path):
        if input('[WARNING] Output directory exists! Overwrite results? ([y]/n): ').lower() == 'n':
            dir_path += '[' + time.strftime('%y-%m-%d_%H%M%S', time.gmtime()) + ']'
            os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)
    print('Results will be saved to %s' % (dir_path))

    return dir_path


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


def visualize_bbox(img, bboxes):
    """
    Plot img and show all bboxes on the img
    
    Args:
        img: PIL image
        bboxes: numpy array or Tensor of size [n_bboxes, 4] 
                i.e. n bboxes each of [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    """
    plt.imshow(img)
    ax = plt.gca()
    for bbox in bboxes:
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red', linewidth=2))
    plt.axis('off')
    plt.show()
    
