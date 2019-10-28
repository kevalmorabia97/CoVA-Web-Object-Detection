import numpy as np
import time
import torch

from utils import print_and_log, print_confusion_matrix


def train_model(model, train_loader, optimizer, criterion, n_epochs, device, eval_loader=None, eval_interval=5, log_file='log.txt'):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    If eval_loader is not None, then evaluate performance on this dataset every `eval_interval` epochs!
    """
    print_and_log('Training Model for %d epochs...' % (n_epochs), log_file)
    model.train()
    for epoch in range(1, n_epochs+1):
        start = time.time()
        epoch_loss, epoch_correct_preds, n_bboxes = 0.0, 0.0, 0.0
        for i, (images, bboxes, labels) in enumerate(train_loader):
            images = images.to(device) # [batch_size, 3, img_H, img_W]
            bboxes = bboxes.to(device) # [total_n_bboxes_in_batch, 5]
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += labels.size()[0]
            
            optimizer.zero_grad()

            output = model(images, bboxes) # [total_n_bboxes_in_batch, n_classes]
            predictions = torch.softmax(output, dim=1).argmax(dim=1)
            epoch_correct_preds += (predictions == labels).sum().item()
            
            loss = criterion(output, labels)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print_and_log('[TRAIN]\t Epoch: %2d\t Loss: %.4f\t Accuracy: %.2f%% (%.2fs)' % (epoch, epoch_loss/n_bboxes, 100*epoch_correct_preds/n_bboxes, time.time()-start), log_file)
        
        if epoch == 1 or epoch%eval_interval == 0 or epoch == n_epochs:
            evaluate_model(model, eval_loader, criterion, device, log_file)
            model.train()
    
    print('Model Trained!')
    return model


def evaluate_model(model, eval_loader, criterion, device, log_file='log.txt'):
    """
    Evaluate `model` (nn.Module) on data loaded by `eval_loader` (torch.utils.data.DataLoader)
    """
    model.eval()
    start = time.time()
    epoch_loss, epoch_correct_preds, n_bboxes = 0.0, 0.0, 0.0
    n_classes = model.n_classes
    class_names = model.class_names
    
    confusion_matrix = np.zeros([n_classes, n_classes], dtype=np.int32) # to get per class metrics
    with torch.no_grad():
        for i, (images, bboxes, labels) in enumerate(eval_loader):
            images = images.to(device) # [batch_size, 3, img_H, img_W]
            bboxes = bboxes.to(device) # [total_n_bboxes_in_batch, 5]
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += labels.size()[0]

            output = model(images, bboxes) # [total_n_bboxes_in_batch, n_classes]
            predictions = torch.softmax(output, dim=1).argmax(dim=1)
            for t, p in zip(labels.view(-1), predictions.view(-1)):
                confusion_matrix[t.item(), p.item()] += 1

            loss = criterion(output, labels)
            epoch_loss += loss.item()
        
        accuracy = confusion_matrix.diagonal().sum()/confusion_matrix.sum()
        per_class_accuracy = confusion_matrix.diagonal()/confusion_matrix.sum(1)
        
        print_and_log('[EVAL]\t Loss: %.4f\t Accuracy: %.2f%% (%.2fs)' % (epoch_loss/n_bboxes, 100*accuracy, time.time()-start), log_file)
        print_confusion_matrix(confusion_matrix, class_names)
        print('')
        for c in range(n_classes):
            print_and_log('%10s Acc: %.2f%%' % (class_names[c], 100*per_class_accuracy[c]), log_file)
        print_and_log('', log_file)
