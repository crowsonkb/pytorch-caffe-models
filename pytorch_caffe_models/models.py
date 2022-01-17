from PIL import Image
import torch
from torch import nn
from torchvision import transforms as T

from . import GoogLeNet


class RGBToBGR(nn.Module):
    """Converts Tensors or PIL images from RGB to BGR (or vice versa)."""

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            if x.ndim not in (3, 4):
                raise ValueError('Tensor must have 3 or 4 dimensions.')
            if x.shape[-3] != 3:
                raise ValueError('Tensor must have three channels.')
            return x.flip(dims=[-3])
        elif isinstance(x, Image.Image):
            x = x.convert('RGB')
            return Image.merge('RGB', x.split()[::-1])
        raise TypeError('Argument must be either Tensor or PIL image.')


def googlenet_bvlc():
    """Returns the model and preprocessing transform for the BVLC GoogLeNet,
    trained on ImageNet.
    
    URL: https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet"""

    transform = T.Compose([
        T.Normalize(mean=[123 / 255, 117 / 255, 104 / 255], std=[1 / 255]),
        RGBToBGR(),
    ])
    model = GoogLeNet(num_classes=1000)
    url = 'https://github.com/crowsonkb/pytorch-caffe-models/releases/download/models-2/bvlc_googlenet-1f25f8c8778a8802.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(url, check_hash=True))
    return model, transform


def googlenet_places205():
    """Returns the model and preprocessing transform for the GoogLeNet
    trained on Places205.

    URL: http://places.csail.mit.edu/downloadCNN.html"""

    transform = T.Compose([
        T.Normalize(mean=[116.047 / 255, 113.753 / 255, 105.417 / 255], std=[1 / 255]),
        RGBToBGR(),
    ])
    model = GoogLeNet(num_classes=205)
    url = 'https://github.com/crowsonkb/pytorch-caffe-models/releases/download/models-2/googlenet_places205-b57a3fc7a34557585.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(url, check_hash=True))
    return model, transform


def googlenet_places365():
    """Returns the model and preprocessing transform for the GoogLeNet
    trained on Places365. The channel-wise means were computed from
    https://github.com/CSAILVision/places365/blob/master/places365CNN_mean.binaryproto.

    URL: https://github.com/CSAILVision/places365"""

    transform = T.Compose([
        T.Normalize(mean=[116.676 / 255, 112.514 / 255, 104.051 / 255], std=[1 / 255]),
        RGBToBGR(),
    ])
    model = GoogLeNet(num_classes=365)
    url = 'https://github.com/crowsonkb/pytorch-caffe-models/releases/download/models-2/googlenet_places365-da1e6512eb7d2613.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(url, check_hash=True))
    return model, transform
