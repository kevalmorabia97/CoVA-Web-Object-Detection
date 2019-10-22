import matplotlib.pyplot as plt
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
    
