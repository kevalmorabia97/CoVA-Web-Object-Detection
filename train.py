import time
import torch


def train_model(model, train_loader, optimizer, criterion, n_epochs, device, eval_loader=None, eval_interval=5):
    """
    Train the `model` (nn.Module) on data loaded by `train_loader` (torch.utils.data.DataLoader) for `n_epochs`.
    If eval_loader is not None, then evaluate performance on this dataset every `eval_interval` epochs!
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()
    for epoch in range(1, n_epochs+1):
        start = time.time()
        epoch_loss = 0.0
        epoch_correct_preds = 0.0
        n_bboxes = 0.0
        for i, (images, bboxes, labels) in enumerate(train_loader):
            images = images.to(device) # [batch_size, 3, img_H, img_W]
            bboxes = bboxes.to(device) # [total_n_bboxes_in_batch, 5]
            labels = labels.to(device) # [total_n_bboxes_in_batch]
            n_bboxes += labels.size()[0]
            
            optimizer.zero_grad()

            predictions = model(images, bboxes) # [total_n_bboxes_in_batch, n_classes]
            predicted_labels = torch.softmax(predictions, dim=1).argmax(dim=1)
            epoch_correct_preds += (predicted_labels == labels).sum().item()
            
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print('[TRAIN]\t Epoch: %2d\t Loss: %.4f\t Accuracy: %.4f (%.2fs)' % (epoch, epoch_loss/n_bboxes, epoch_correct_preds/n_bboxes, time.time()-start))
        
        if epoch == 1 or epoch%eval_interval == 0 or epoch == n_epochs:
            evaluate_model(model, eval_loader, optimizer, criterion, device)
            model.train()
    
    print('Model Trained!')
    return model


def evaluate_model(model, eval_loader, optimizer, criterion, device):
    """
    Evaluate `model` (nn.Module) on data loaded by `eval_loader` (torch.utils.data.DataLoader)
    """
    model.eval()
    epoch_loss = 0.0
    epoch_correct_preds = 0.0
    n_bboxes = 0.0
    for i, (images, bboxes, labels) in enumerate(eval_loader):
        images = images.to(device) # [batch_size, 3, img_H, img_W]
        bboxes = bboxes.to(device) # [total_n_bboxes_in_batch, 5]
        labels = labels.to(device) # [total_n_bboxes_in_batch]
        n_bboxes += labels.size()[0]

        optimizer.zero_grad()

        predictions = model(images, bboxes) # [total_n_bboxes_in_batch, n_classes]
        predicted_labels = torch.softmax(predictions, dim=1).argmax(dim=1)
        epoch_correct_preds += (predicted_labels == labels).sum().item()

        loss = criterion(predictions, labels)
        epoch_loss += loss.item()

    print('[EVAL]\t Loss: %.4f\t Accuracy: %.4f' % (epoch_loss/n_bboxes, epoch_correct_preds/n_bboxes))