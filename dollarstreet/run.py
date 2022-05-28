import time
import copy

import torch
import torch.backends.cudnn as cudnn


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    cudnn.benchmark = True
else:
    device = torch.device('cpu')


def _run_epoch(model, optimizer, criterion, dataloaders, phase):
    dataloader = dataloaders[phase]

    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        # Save batch stats
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    print(f'({phase}) Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model, epoch_acc


def _run_epochs(model, dataloaders, criterion, optimizer, num_epochs, train):
    phases = ['train', 'val'] if train else ['val']
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Iterate over epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Iterate through training and validation phase
        for phase in phases:
            model, epoch_acc = _run_epoch(
                model, optimizer, criterion, dataloaders, phase)

            # Save model if it has better performance
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # Save epoch stats
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    return _run_epochs(
        model, dataloaders, criterion, optimizer, num_epochs, True)


def validate_model(model, dataloaders, criterion, optimizer):
    return _run_epochs(
        model, dataloaders, criterion, optimizer, 1, False)
