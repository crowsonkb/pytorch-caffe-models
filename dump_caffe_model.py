#!/usr/bin/env python3

"""Dumps a Caffe binary model to a pickle of NumPy arrays."""

import argparse
from collections import OrderedDict
import os
import pickle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=str,
                   help='the model definition (.prototxt)')
    p.add_argument('weights', type=str,
                   help='the model weights (.caffemodel)')
    p.add_argument('output', type=str,
                   help='the output file')
    args = p.parse_args()

    os.environ['GLOG_minloglevel'] = '1'
    import caffe

    caffe.set_mode_cpu()
    net = caffe.Net(args.model, 1, weights=args.weights)

    out = OrderedDict()

    for name, layer in net.layer_dict.items():
        out[name] = [blob.data for blob in layer.blobs]
        for i, blob in enumerate(layer.blobs):
            print(f'Found layer {name}, blob {i}, shape {blob.data.shape}')

    with open(args.output, 'wb') as fp:
        pickle.dump(out, fp)
    print(f'Written to {args.output}.')


if __name__ == '__main__':
    main()
