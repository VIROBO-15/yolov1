import torch
import numpy as np


IMG_WIDTH = 448
IMG_HEIGHT = 448
S = 7   # number of grid cell is S*S
B = 2   # number of bbox for each grid cell
C = 20  # number of classses

def read_labels(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
        labels = []
        for l in lines:
            l = l.split()
            l = [float(elem) for elem in l]
            labels.append(l)
    return labels

def labels2tensor(labels):
    """
    Build Groundtruth tensor S*S*5.
    :param labels: list of labels with bounding box classification and position for each image.
    :return: T: Groundtruth tensor S*S*5.
                format <x> <y> <w> <h> <class name>
    """
    T = torch.zeros(S, S, 5)  # init

    gcell_size = 1. / S
    for label in labels:  # mark labels
        cls = label[0]
        x = label[1]
        y = label[2]
        w = label[3]
        h = label[4]
        # Be aware: row are x-axis image coordinate, in 2nd dimension of Tensor

        T[int(y/gcell_size), int(x/gcell_size), 0] = x
        T[int(y/gcell_size), int(x/gcell_size), 1] = y
        T[int(y/gcell_size), int(x/gcell_size), 2] = w
        T[int(y/gcell_size), int(x/gcell_size), 3] = h
        T[int(y/gcell_size), int(x/gcell_size), 4] = cls

        '''
        # w,h already related to whole image, no action required
        # normalize x,y to grid cell offset
        x = (x - int(x/gcell_size) * gcell_size) / gcell_size
        y = (y - int(y/gcell_size) * gcell_size) / gcell_size
        '''
        T[int(y / gcell_size), int(x / gcell_size)] = torch.tensor([x, y, w, h, cls])

    return T