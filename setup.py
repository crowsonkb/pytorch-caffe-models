import setuptools

setuptools.setup(
    name='pytorch-caffe-models',
    version='0.1',
    description='The original weights of some Caffe models, ported to PyTorch.',
    # TODO: add long description
    long_description='The original weights of some Caffe models, ported to PyTorch.',
    url='https://github.com/crowsonkb/pytorch-caffe-models',
    author='Katherine Crowson',
    author_email='crowsonkb@gmail.com',
    license='MIT',
    packages=['pytorch_caffe_models'],
    install_requires=['Pillow',
                      'torch',
                      'torchvision',
                      'tqdm'],
    python_requires=">=3.6",
    # TODO: Add classifiers
    classifiers=[],
)
