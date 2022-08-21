import copy
import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader

from dollarstreet import device
import dollarstreet.constants as c
from dollarstreet.models import get_model
from dollarstreet.utils import AverageMeter, log


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    #import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def _accuracy(output, target, topk=(1,)) -> List[torch.Tensor]:
    """Computes the precision@k for the specified values of k.
    """
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


def _run_epochs(
        model_names,
        dataloaders,
        num_epochs,
        seed,
        train) -> Tuple[list, dict, int, int, bool]:
    """Performs training and/or validation. Returns best models and scores.
    """

    seed_everything(seed=seed)
    # Initialize pytorch objects
    models = {name: get_model(name) for name in model_names}
    optimizers = {
        name: torch.optim.SGD(
            models[name].parameters(),
            c.LR,
            c.MOMENTUM,
            weight_decay=c.WEIGHT_DECAY)
        for name in model_names
    }
    criterions = {name: nn.CrossEntropyLoss() for name in model_names}

    # Initialize data stores
    top1_history = {name: [] for name in model_names}
    top5_history = {name: [] for name in model_names}
    best_model_wts = {
        name: copy.deepcopy(models[name].state_dict())
        for name in model_names
    }
    best_top1 = {name: 0.0 for name in model_names}

    # Iterate over epochs
    since = time.time()
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        logger.info('-' * 10)

        # Initialize counters
        losses = {name: AverageMeter() for name in model_names}
        top1 = {name: AverageMeter() for name in model_names}
        top5 = {name: AverageMeter() for name in model_names}

        # Iterate through training and validation phase
        phases = ['train', 'val'] if train else ['val']
        for phase in phases:
            dataloader = dataloaders[phase]

            if phase == 'train':
                for name, model in models.items():
                    models[name] = model.train()
            else:
                for name, model in models.items():
                    models[name] = model.eval()

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                for name in model_names:
                    optimizers[name].zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = models[name](inputs)
                        loss = criterions[name](outputs, labels)

                        prec1, prec5 = _accuracy(
                            outputs.data, labels, topk=(1, 5))
                        losses[name].update(loss.data.item(), inputs.size(0))
                        top1[name].update(prec1[0], inputs.size(0))
                        top5[name].update(prec5[0], inputs.size(0))

                        if phase == 'train':
                            loss.backward()
                            optimizers[name].step()

            for name in models:
                logger.info(
                    f'({name},{phase}) '
                    f'Loss: {losses[name].avg:.4f} '
                    f'prec@1: {top1[name].avg:.4f} '
                    f'prec@5: {top5[name].avg:.4f}'
                )

                # Save model if it has better performance
                if phase == 'val' and top1[name].avg > best_top1[name]:
                    best_top1[name] = top1[name].avg
                    best_model_wts[name] = copy.deepcopy(
                        models[name].state_dict())

                # Save epoch stats
                if phase == 'val':
                    top1_history[name].append(top1[name].avg)
                    top5_history[name].append(top5[name].avg)

    time_elapsed = time.time() - since
    logger.info(
        f'Complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    logger.info('Best prec@1:')

    for name in model_names:
        logger.info(f'({name}) {best_top1[name]:4f}')

        # load best model weights
        models[name].load_state_dict(best_model_wts[name])

    return models, top1_history, top5_history


def _run_epoch(
        model,
        dataloaders,
        num_epochs,
        seed,
        train) -> Tuple[Module, list, list]:
    """Performs training and/or validation. Returns best models and scores.
    """
    
    seed_everything(seed=seed)
    # Initialize pytorch objects
    optimizer = torch.optim.SGD(
        model.parameters(),
        c.LR,
        c.MOMENTUM,
        weight_decay=c.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Initialize data stores
    top1_history = []
    top5_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_top1 = 0.0

    # Iterate over epochs
    since = time.time()
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        logger.info('-' * 10)

        # Initialize counters
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # Iterate through training and validation phase
        phases = ['train', 'val'] if train else ['val']
        for phase in phases:
            dataloader = dataloaders[phase]

            if phase == 'train':
                model = model.train()
            else:
                models = model.eval()

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    prec1, prec5 = _accuracy(
                        outputs.data, labels, topk=(1, 5))
                    losses.update(loss.data.item(), inputs.size(0))
                    top1.update(prec1[0], inputs.size(0))
                    top5.update(prec5[0], inputs.size(0))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            logger.info(
                f'({phase}) '
                f'Loss: {losses.avg:.4f} '
                f'prec@1: {top1.avg:.4f} '
                f'prec@5: {top5.avg:.4f}'
            )

            # Save model if it has better performance
            if phase == 'val' and top1.avg > best_top1:
                best_top1 = top1.avg
                best_model_wts = copy.deepcopy(
                    models.state_dict())

            # Save epoch stats
            if phase == 'val':
                top1_history.append(top1.avg)
                top5_history.append(top5.avg)

    time_elapsed = time.time() - since
    logger.info(
        f'Complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    logger.info(f'Best prec@1: {best_top1:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, top1_history, top5_history


@log(logger=logger, log_output=False)
def train_models(
        model_names: List[str],
        dataloaders: Dict[str, DataLoader],
        num_epochs: int,
        seed: int,
        save_log: Optional[bool] = False,
        description: Optional[str] = None) -> Tuple[list, dict, int, int, bool, str]:
    """Trains models. Return models and scores.

    Args:
        model_names (List[str]): List of valid model names.
        dataloaders (Dict[str, DataLoader]): Dictionary of dataloaders
            for train and val phases.
        num_epochs (int): Number of epochs for training.
        save_log (optional, bool): Flag for log decorator. When true, logs
            are saved to disk.
        description (optional, str): Description for logging.

    Returns:
        Tuple[dict, dict, dict]: models, top1 scores, top5 scores
    """

    return _run_epochs(
        model_names, dataloaders, num_epochs, seed, True)


@log(logger=logger, log_output=False)
def validate_models(
        model_names: List[str],
        dataloaders: Dict[str, DataLoader],
        seed: int,
        save_log: Optional[bool] = False,
        description: Optional[str] = None) -> Tuple[dict, dict, dict]:
    """Validates models. Return models and scores.

    Args:
        model_names (List[str]): List of valid model names.
        dataloaders (Dict[str, DataLoader]): Dictionary of dataloaders
            for train and val phases.
        save_log (optional, bool): Flag for log decorator. When true, logs
            are saved to disk.
        description (optional, str): Description for logging.

    Returns:
        Tuple[dict, dict, dict]: models, top1 scores, top5 scores
    """

    return _run_epochs(
        model_names, dataloaders, 1, seed, False)


@log(logger=logger, log_output=False)
def validate_model(
        dataloaders: Dict[str, DataLoader],
        seed: int,
        model: Optional[Module] = None,
        model_name: Optional[str] = None,
        save_log: Optional[bool] = False,
        description: Optional[str] = None) -> Tuple[Module, list, list]:
    """Validates a single model. Return scores.

    Args:
        dataloaders (Dict[str, DataLoader]): Dictionary of dataloaders
            for train and val phases.
        model (optiomal, Module): Torch model.
        model_name (optional, str): Valid model name.
        save_log (optional, bool): Flag for log decorator. When true, logs
            are saved to disk.
        description (optional, str): Description for logging.

    Returns:
        Tuple[Module, list, list]: model, top1 scores, top5 scores
    """

    # If no model is provided, load model
    if model is None:
        model = get_model(model_name)

    return _run_epoch(
        model=model, dataloaders=dataloaders, num_epochs=1, seed=seed, train=False)
