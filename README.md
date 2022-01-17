# pytorch-caffe-models

This repo contains the original weights of some Caffe models, ported to PyTorch. Currently there are:

GoogLeNet ([Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)):

* [BVLC GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet), trained on ImageNet.

* [GoogLeNet trained on Places205](http://places.csail.mit.edu/downloadCNN.html).

* [GoogLeNet trained on Places365](https://github.com/CSAILVision/places365).

The [GoogLeNet model in torchvision](https://pytorch.org/vision/stable/models.html#torchvision.models.googlenet) was trained from scratch by the PyTorch team with very different data preprocessing and has very differently scaled internal activations, which can be important when using the model as a feature extractor.

There is also a tool (`dump_caffe_model.py`) to dump Caffe model weights to a more portable format (pickles of NumPy arrays), which requires Caffe and its Python 3 bindings to be installed. A script to compute validation loss and accuracy (`validate.py`) is also included (the ImageNet validation set can be obtained [from Academic Torrents](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5)).

## Usage

### Basic usage

This outputs logits for 1000 ImageNet classes for a black (zero) input image:

```python
import pytorch_caffe_models

model, transform = pytorch_caffe_models.googlenet_bvlc()

model(transform(torch.zeros([1, 3, 224, 224])))
```

The original models were trained with BGR input data in the range 0-255, which had then been scaled to zero mean but not unit standard deviation. The model-specific transform returned by the pretrained model creation function expects RGB input data in the range 0-1 and it will differentiably rescale the input and convert from RGB to BGR.

### Feature extraction

Using the new [torchvision feature extraction utility](https://pytorch.org/vision/stable/feature_extraction.html):

```python
from torchvision.models import feature_extraction

layer_names = feature_extraction.get_graph_node_names(model)[1]
```

Then pick your favorite layer (we can use `inception_4c.conv_5x5`)

```python
model.eval().requires_grad_(False)
extractor = feature_extraction.create_feature_extractor(model, {'inception_4c.conv_5x5': 'out'})

input_image = torch.randn([1, 3, 224, 224]) / 50 + 0.5
input_image.requires_grad_()

features = extractor(transform(input_image))['out']
loss = -torch.sum(features**2) / 2
loss.backward()
```

`input_image` now has its `.grad` attribute populated and you can normalize and descend this gradient for [DeepDream](https://en.wikipedia.org/wiki/DeepDream) or other feature visualization methods. (The BVLC GoogLeNet model was the most popular model used for DeepDream.)
