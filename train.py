import numpy as np
import time
import torch

from utils import print_and_log, print_confusion_matrix


def train_model(model, train_loader, optimizer, criterion, n_epochs, device, eval_loader, eval_interval=3, log_file='log.txt', ckpt_path='ckpt.pth'):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    evaluate performance on `eval_loader` dataset every `eval_interval` epochs and check for early stopping criteria!
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()

    best_eval_acc = 0.0
    patience = 5 # number of VAL Acc values observed after best value to stop training
    min_delta = 1e-5 # min improvement in eval_acc value to be considered a valid improvement
    for epoch in range(1, n_epochs+1):
        start = time.time()
        epoch_loss, epoch_correct_preds, n_bboxes = 0.0, 0.0, 0.0
        for i, (images, bboxes, context_indices, labels) in enumerate(train_loader):
            images = images.to(device) # [batch_size, 3, img_H, img_W]
            bboxes = bboxes.to(device) # [total_n_bboxes_in_batch, 5]
            context_indices = context_indices.to(device) # [total_n_bboxes_in_batch, 2 * context_size]
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += labels.size()[0]
            
            optimizer.zero_grad()

            output = model(images, bboxes, context_indices) # [total_n_bboxes_in_batch, n_classes]
            predictions = output.argmax(dim=1)
            epoch_correct_preds += (predictions == labels).sum().item()
            
            loss = criterion(output, labels)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print_and_log('[TRAIN]\t Epoch: %2d\t Loss: %.4f\t Accuracy: %.2f%% (%.2fs)' % (epoch, epoch_loss/n_bboxes, 100*epoch_correct_preds/n_bboxes, time.time()-start), log_file)
        
        if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
            per_class_accuracy = evaluate_model(model, eval_loader, criterion, device, 'VAL', log_file)
            eval_acc = per_class_accuracy[1:].mean()
            model.train()

            if eval_acc - best_eval_acc > min_delta: # best so far so save checkpoint to restore later
                best_eval_acc = eval_acc
                patience_count = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_count += 1
                if patience_count >= patience:
                    print('Early Stopping!')
                    break
    
    print('Model Trained! Restoring model to best Eval performance checkpoint...')
    model.load_state_dict(torch.load(ckpt_path))


def evaluate_model(model, eval_loader, criterion, device, split_name='VAL', log_file='log.txt'):
    """
    Evaluate model (nn.Module) on data loaded by eval_loader (torch.utils.data.DataLoader)
    eval_loader.batch_size SHOULD BE 1
    
    Returns: per_class_accuracy np.array of shape [n_classes,]
    """
    assert eval_loader.batch_size == 1
    
    model.eval()
    start = time.time()
    epoch_loss, epoch_correct_preds, n_bboxes = 0.0, 0.0, 0.0
    n_classes = model.n_classes
    class_names = model.class_names
    confusion_matrix = np.zeros([n_classes, n_classes], dtype=np.int32) # to get per class metrics
    with torch.no_grad():
        for i, (images, bboxes, context_indices, labels) in enumerate(eval_loader):
            images = images.to(device) # [batch_size, 3, img_H, img_W]
            bboxes = bboxes.to(device) # [total_n_bboxes_in_batch, 5]
            context_indices = context_indices.to(device) # [total_n_bboxes_in_batch, 2 * context_size]
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += labels.size()[0]
            output = model(images, bboxes, context_indices) # [total_n_bboxes_in_batch, n_classes]
            
            price_bb = output[:, 1].argmax()
            image_bb = output[:, 2].argmax()
            title_bb = output[:, 3].argmax()
            labels_flattened = labels.view(-1)
            for j, l in enumerate(labels_flattened):
                l = l.item()
                if l > 0:
                    if j == price_bb:
                        confusion_matrix[l, 1] += 1
                    elif j == image_bb:
                        confusion_matrix[l, 2] += 1
                    elif j == title_bb:
                        confusion_matrix[l, 3] += 1
                    else:
                        confusion_matrix[l, 0] += 1
            
            if labels_flattened[price_bb].item() == 0:
                confusion_matrix[0, 1] += 1
            if labels_flattened[image_bb].item() == 0:
                confusion_matrix[0, 2] += 1
            if labels_flattened[title_bb].item() == 0:
                confusion_matrix[0, 3] += 1
            
            loss = criterion(output, labels)
            epoch_loss += loss.item()
        
        per_class_accuracy = confusion_matrix.diagonal()/confusion_matrix.sum(1)
        avg_accuracy = per_class_accuracy[1:].mean() # accuracy of classes other than BG
        print_and_log('[%s]\t Loss: %.4f\t Avg_class_Accuracy: %.2f%% (%.2fs)' % (split_name, epoch_loss/n_bboxes, 100*avg_accuracy, time.time()-start), log_file)
        # print_confusion_matrix(confusion_matrix, class_names)
        for c in range(1, n_classes):
            print_and_log('%-5s Acc: %.2f%%' % (class_names[c], 100*per_class_accuracy[c]), log_file)
        print_and_log('', log_file)
        
        return per_class_accuracy
