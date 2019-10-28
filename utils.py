import matplotlib.pyplot as plt
import os
import pickle
import torch


def accuracy(predictions, labels):
    """
    Compute accuracy of predictions
    
    Args:
        predictions: raw prediction scores as a float32 torch.Tensor of size [n_examples, n_classes]
        labels: actual class labels as long torch.Tensor of size [n_examples]
    """
    predicted_labels = torch.softmax(predictions, dim=1).argmax(dim=1)
    corrects = (predicted_labels == labels).sum().float()
    accuracy = corrects / float( labels.size(0) )
    
    return accuracy


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


def count_parameters(model):
    """
    Return the number of trainable parameters in `model`
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def custom_collate_fn(batch):
    """
    Since all images might have different number of BBoxes, to use batch_size > 1,
    custom collate_fn has to be created that creates a batch
    
    Args:
        batch: list of N=`batch_size` tuples. Example [(img_1, bboxes_1, labels_1), ..., (img_N, bboxes_N, labels_N)]
    
    Returns:
        batch: contains images, bboxes, labels
            images: torch.Tensor [N, 3, img_H, img_W]
            bboxes: torch.Tensor [total_n_bboxes_in_batch, 5]
                each each of [bath_img_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            labels: torch.Tensor [total_n_bboxes_in_batch]
    """
    images, bboxes, labels = zip(*batch)
    # images = (img_1, ..., img_N) each element of size [3, img_H, img_W]
    # bboxes = (bboxes_1, ..., bboxes_N) each element of size [n_bboxes_in_image, 4]
    # labels = (labels_1, ..., labels_N) each element of size [n_bboxes_in_image]
    
    images = torch.stack(images, 0)
    
    bboxes_with_batch_index = []
    for i, bbox in enumerate(bboxes):
        batch_indices = torch.Tensor([i]*bbox.size()[0]).view(-1,1)
        bboxes_with_batch_index.append(torch.cat((batch_indices, bbox), dim=1))
    bboxes_with_batch_index = torch.cat(bboxes_with_batch_index)
    
    labels = torch.cat(labels)
    
    return images, bboxes_with_batch_index, labels


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
    
