import torch


def train_model(model, dataloader, optimizer, criterion, n_epochs, device):
    """
    Train the `model` (nn.Module) on data loaded by `dataloader` (torch.utils.data.DataLoader) for `n_epochs`
    """
    print('Training Model for %d epochs...' % (n_epochs))
    model.train()
    for epoch in range(1, n_epochs+1):
        epoch_loss = 0.0
        epoch_correct_preds = 0.0
        n_bboxes = 0.0
        for i, (images, bboxes, labels) in enumerate(dataloader):
            if bboxes.size() == torch.Size([1,0]): # No BBoxes in this image
                continue
            
            bboxes = bboxes[0]
            
            images = images.to(device) # [batch_size, 3, img_H, img_W]
            bboxes = bboxes.to(device) # [n_bboxes, 4]
            labels = labels.squeeze(0).to(device) # [n_bboxes]
            n_bboxes += labels[0]
            
            optimizer.zero_grad()

            predictions = model(images, [bboxes]) # [n_bboxes, n_classes]
            predicted_labels = torch.softmax(predictions, dim=1).argmax(dim=1)
            epoch_correct_preds += (predicted_labels == labels).sum().item()
            
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print('Epoch: %2d\t Loss: %.4f\t Accuracy: %.4f' % (epoch, epoch_loss/n_bboxes, epoch_correct_preds/n_bboxes))
    
    print('Model Trained!')
    return model