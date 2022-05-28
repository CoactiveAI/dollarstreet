import time
import copy

import torch
import torch.backends.cudnn as cudnn

from dollarstreet.utils import AverageMeter


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    cudnn.benchmark = True
else:
    device = torch.device('cpu')


def _accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def _run_epochs(model, dataloaders, criterion, optimizer, num_epochs, train):
    phases = ['train', 'val'] if train else ['val']

    top1_history = []
    top5_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_top1 = 0.0

    # Iterate over epochs
    since = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Initialize counters
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # Iterate through training and validation phase
        for phase in phases:
            dataloader = dataloaders[phase]

            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    prec1, prec5 = _accuracy(outputs.data, labels, topk=(1, 5))
                    losses.update(loss.data.item(), inputs.size(0))
                    top1.update(prec1[0], inputs.size(0))
                    top5.update(prec5[0], inputs.size(0))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            print(
                f'({phase}) Loss: {losses.avg:.4f} prec@1: {top1.avg:.4f} prec@5: {top5.avg:.4f}')

            # Save model if it has better performance
            if phase == 'val' and top1.avg > best_top1:
                best_top1 = top1.avg
                best_model_wts = copy.deepcopy(model.state_dict())

            # Save epoch stats
            if phase == 'val':
                top1_history.append(top1.avg)
                top5_history.append(top5.avg)

        print()

    time_elapsed = time.time() - since
    print(
        f'Complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best prec@1: {best_top1:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, top1_history, top5_history


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    return _run_epochs(
        model, dataloaders, criterion, optimizer, num_epochs, True)


def validate_model(model, dataloaders, criterion, optimizer):
    return _run_epochs(
        model, dataloaders, criterion, optimizer, 1, False)
