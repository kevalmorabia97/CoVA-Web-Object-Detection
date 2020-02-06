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
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += labels.shape[0]
            
            optimizer.zero_grad()

            output = model(images.to(device), bboxes.to(device), context_indices.to(device)) # [total_n_bboxes_in_batch, n_classes]
            predictions = output.argmax(dim=1)
            epoch_correct_preds += (predictions == labels).sum().item()
            
            loss = criterion(output, labels)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print_and_log('[TRAIN]\t Epoch: %2d\t Loss: %.4f\t Accuracy: %.2f%% (%.2fs)' % (epoch, epoch_loss/n_bboxes, 100*epoch_correct_preds/n_bboxes, time.time()-start), log_file)
        
        if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
            class_acc = evaluate_model(model, eval_loader, criterion, device, 1, 'VAL', log_file)
            eval_acc = class_acc[1:].mean()
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

    return best_eval_acc


def evaluate_model(model, eval_loader, criterion, device, k=1, split_name='VAL', log_file='log.txt'):
    """
    Evaluate model (nn.Module) on data loaded by eval_loader (torch.utils.data.DataLoader)
    Check top `k` (default: 1) predictions for each class while evaluating class accuracies
    Returns: class_acc np.array of shape [n_classes,]
    """
    start = time.time()
    
    model.eval()
    epoch_loss, n_bboxes = 0.0, 0.0
    n_classes = model.n_classes
    confusion_matrix = np.zeros([n_classes, n_classes], dtype=np.int32) # to get per class metrics
    with torch.no_grad():
        for i, (images, bboxes, context_indices, labels) in enumerate(eval_loader):
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += labels.shape[0]

            output = model(images.to(device), bboxes.to(device), context_indices.to(device)) # [total_n_bboxes_in_batch, n_classes]
            loss = criterion(output, labels)
            epoch_loss += loss.item()
            
            batch_indices = torch.unique(bboxes[:,0])
            for index in batch_indices: # for each image
                img_indices = (bboxes[:,0] == index)
                labels_img = labels[img_indices].view(-1,1)
                output_img = output[img_indices]

                label_indices = torch.arange(labels_img.shape[0], device=device).view(-1,1)
                indexed_labels = torch.cat((label_indices, labels_img), dim=1)
                indexed_labels = indexed_labels[indexed_labels[:,-1] != 0] # labels for bbox other than BG
                
                top_k_predictions = torch.argsort(output_img, dim=0)[output_img.shape[0]-k:] # [k, n_classes] indices indicating top k predicted bbox
                for c in range(1, n_classes):
                    true_bbox = indexed_labels[indexed_labels[:,-1] == c][0,0]
                    pred_bboxes = top_k_predictions[:, c]
                    confusion_matrix[c, c if true_bbox in pred_bboxes else 0] += 1

        confusion_matrix[0, 0] += 1 # to avoid div by 0
        class_acc = confusion_matrix.diagonal()/confusion_matrix.sum(1)
        avg_acc = class_acc[1:].mean() # accuracy of classes other than BG
        
        print_and_log('[%s]\t Loss: %.4f\t Avg_class_Accuracy: %.2f%% (%.2fs)' % (split_name, epoch_loss/n_bboxes, 100*avg_acc, time.time()-start), log_file)
        for c in range(1, n_classes):
            print_and_log('%s top-%d-Acc: %.2f%%' % (model.class_names[c], k, 100*class_acc[c]), log_file)
        print_and_log('', log_file)
        
        return class_acc