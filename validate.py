#!/usr/bin/env python3

"""Reports GoogLeNet validation loss and accuracy on ImageNet or other datasets."""

import argparse

import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, models, transforms as T
from tqdm import tqdm

import pytorch_caffe_models


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=str, required=True,
                   help='the root of the ImageNet validation set')
    p.add_argument('--model-type', type=str, required=True,
                   choices=['googlenet_bvlc', 'googlenet_places205', 'googlenet_places365'])
    p.add_argument('--batch-size', type=int, default=128,
                   help='the batch size')
    p.add_argument('--num-workers', type=int, default=16,
                   help='the number of data loader worker processes')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    if args.model_type == 'googlenet_bvlc':
        model, preprocess = pytorch_caffe_models.googlenet_bvlc()
    elif args.model_type == 'googlenet_places205':
        model, preprocess = pytorch_caffe_models.googlenet_places205()
    elif args.model_type == 'googlenet_places365':
         model, preprocess = pytorch_caffe_models.googlenet_places365()
    else:
        raise ValueError('Invalid model type.')

    model = model.to(device).eval().requires_grad_(False)

    tf = T.Compose([
        T.Resize((256, 256), T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        preprocess,
    ])
    dataset = datasets.ImageFolder(args.dataset, transform=tf)
    dataloader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers)

    count = 0
    total_loss = 0.
    total_top_1 = 0
    total_top_5 = 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    torch.manual_seed(3393410851)

    with torch.inference_mode():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels)
            _, top_5 = logits.topk(5)
            total_top_1 += torch.sum(top_5[:, 0] == labels)
            total_top_5 += torch.sum(top_5 == labels[:, None])
            count += len(images)
    
    loss = total_loss / count
    accuracy_top_1 = total_top_1 / count
    accuracy_top_5 = total_top_5 / count

    print(f'Validation loss: {loss:g}, top 1 accuracy: {accuracy_top_1:g}, '
          f'top 5 accuracy: {accuracy_top_5:g}')


if __name__ == '__main__':
    main()
