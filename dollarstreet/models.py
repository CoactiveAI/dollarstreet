from typing import Optional

from torch.nn import Module
from torchvision import models

from dollarstreet import device
import dollarstreet.constants as c


def get_model(
        model_name: str, use_pretrained: Optional[bool] = True) -> Module:
    """Return specified pytorch model.

    Args:
        model_name (str): Name of model to get.
        use_pretrained (Optional[bool], optional): Pretrained flag.

    Returns:
        Module: Pytorch model.
    """
    assert model_name in c.VALID_MODELS, f'{model_name}: Invalid model name'

    if model_name == "resnet":
        model = models.resnet18(pretrained=use_pretrained)

    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=use_pretrained)

    elif model_name == "densenet":
        model = models.densenet121(pretrained=use_pretrained)

    elif model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=use_pretrained)

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(pretrained=use_pretrained)

    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x1_0(pretrained=use_pretrained)

    elif model_name == "visionnet":
        model = models.vit_b_16(pretrained=use_pretrained)

    # TODO: Add to constants.py the valid ones

    model = model.to(device)
    return model
