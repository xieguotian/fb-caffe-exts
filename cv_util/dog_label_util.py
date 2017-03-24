import numpy as np
from collections import OrderedDict

def get_dog_label_map(all_label_file,dog_label_file):
    imagenet_1k = OrderedDict()
    with open(all_label_file) as fid:
        for line in fid:
            str = line.split()
            key = str[0].split('/')[0].strip()
            label = np.int(str[1])

            imagenet_1k[key] = label

    dog_120 = OrderedDict()
    with open(dog_label_file) as fid:
        for line in fid:
            str = line.split()
            key = str[0].split('/')[0].strip()
            label = np.int(str[1])
            dog_120[imagenet_1k[key]] = label
    return dog_120

