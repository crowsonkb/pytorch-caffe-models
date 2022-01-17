# pytorch-caffe-models

This repo contains the original weights of some Caffe models, ported to PyTorch. Currently there are:

GoogLeNet ([Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)):

* [BVLC GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet), trained on ImageNet.

* [GoogLeNet trained on Places205](http://places.csail.mit.edu/downloadCNN.html).

* [GoogLeNet trained on Places365](https://github.com/CSAILVision/places365).

The [GoogLeNet model in torchvision](https://pytorch.org/vision/stable/models.html#torchvision.models.googlenet) was trained from scratch by the PyTorch team with very different data preprocessing and has very differently scaled internal activations, which can be important when using the model as a feature extractor.

There is also a tool (`dump_caffe_model.py`) to dump Caffe model weights to a more portable format (pickles of NumPy arrays), which requires Caffe and its Python 3 bindings to be installed. A script to compute validation loss and accuracy (`validate.py`) is also included (the ImageNet validation set can be obtained [from Academic Torrents](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5)).
