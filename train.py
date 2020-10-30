import numpy as np
from time import time
import torch

from utils import print_and_log


def train_model(model, train_loader, optimizer, scheduler, criterion, n_epochs, device, eval_loader, eval_interval=3, log_file='log.txt',
                model_save_file='ckpt.pth'):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    evaluate performance on `eval_loader` dataset every `eval_interval` epochs and check for early stopping criteria!
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()

    best_eval_acc = 0.0
    patience = 10 # number of VAL Acc values observed after best value to stop training
    for epoch in range(1, n_epochs+1):
        start = time()
        epoch_loss, epoch_correct, n_bboxes = 0.0, 0.0, 0.0
        for _, images, bboxes, additional_features, context_indices, labels in train_loader:
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += labels.shape[0]
            
            optimizer.zero_grad()

            output = model(images.to(device), bboxes.to(device), additional_features.to(device), context_indices.to(device)) # [total_n_bboxes_in_batch, n_classes]
            predictions = output.argmax(dim=1) # [total_n_bboxes_in_batch]
            epoch_correct += (predictions == labels).sum().item()
            
            loss = criterion(output, labels)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print_and_log('Epoch: %2d  Loss: %.4f  Accuracy: %.2f%%  (%.2fs)' % (epoch, epoch_loss/n_bboxes, 100*epoch_correct/n_bboxes, time()-start), log_file)
        if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
            _, class_acc = evaluate_model(model, eval_loader, device, 1, 'VAL', log_file)
            eval_acc = class_acc.mean()
            model.train()

            if eval_acc > best_eval_acc: # best so far so save checkpoint to restore later
                best_eval_acc = eval_acc
                patience_count = 0
                torch.save(model.state_dict(), model_save_file)
            else:
                patience_count += 1
                if patience_count >= patience:
                    print('Early Stopping!')
                    break
        
        scheduler.step()
    
    print('Model Trained! Restoring model to best Eval performance checkpoint...')
    model.load_state_dict(torch.load(model_save_file))

    return best_eval_acc


@torch.no_grad()
def evaluate_model(model, eval_loader, device, k=1, split_name='VAL', log_file='log.txt'):
    """
    Evaluate model (nn.Module) on data loaded by eval_loader (torch.utils.data.DataLoader)
    Check top `k` (default: 1) predictions for each class while evaluating class accuracies
    Returns:
        `img_acc`: np.array (np.int32) of shape [n_imgs, 4], each row contains [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
        `class_acc`: of classes other than BG, np.array of shape [n_classes-1,] where values are in percentages
    """
    start = time()
    
    model.eval()
    n_classes = model.n_classes
    img_acc = [] # list of [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
    for img_ids, images, bboxes, additional_features, context_indices, labels in eval_loader:
        labels = labels.to(device) # [total_n_bboxes_in_batch]
        output = model(images.to(device), bboxes.to(device), additional_features.to(device), context_indices.to(device)) # [total_n_bboxes_in_batch, n_classes]
        
        batch_indices = torch.unique(bboxes[:,0]).long()
        for index in batch_indices: # for each image
            img_id = img_ids[index].item()
            img_indices = (bboxes[:,0] == index)
            labels_img = labels[img_indices].view(-1,1)
            output_img = output[img_indices]

            label_indices = torch.arange(labels_img.shape[0], device=device).view(-1,1)
            indexed_labels = torch.cat((label_indices, labels_img), dim=1)
            indexed_labels = indexed_labels[indexed_labels[:,-1] != 0] # labels for bbox other than BG
            
            top_k_predictions = torch.argsort(output_img, dim=0)[output_img.shape[0]-k:] # [k, n_classes] indices indicating top k predicted bbox
            curr_img_acc = [img_id] # [img_id, price_acc (1/0), title_acc (1/0), image_acc (1/0)]
            for c in range(1, n_classes):
                true_bbox = indexed_labels[indexed_labels[:,-1] == c][0,0]
                pred_bboxes = top_k_predictions[:, c]
                curr_img_acc.append(1 if true_bbox in pred_bboxes else 0)
            img_acc.append(curr_img_acc)
        
    img_acc = np.array(img_acc, dtype=np.int32) # [n_imgs, 4] numpy array
    class_acc = img_acc[:,1:].mean(0)*100 # accuracies of classes other than BG
    
    print_and_log('[%s] Avg_class_Accuracy: %.2f%% (%.2fs)' % (split_name, class_acc.mean(), time()-start), log_file)
    for c in range(1, n_classes):
        print_and_log('%s top-%d-Acc: %.2f%%' % (model.class_names[c], k, class_acc[c-1]), log_file)
    print_and_log('', log_file)
        
    return img_acc, class_acc