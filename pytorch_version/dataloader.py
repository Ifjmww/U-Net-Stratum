from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
from PIL import Image
import numpy as np
import torch
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt


class UNetDataset(Dataset):
    def __init__(self, path, data_name, batchsize=1, steps=None, shuffle=False, transforms=None):
        self.x, self.y = self.load_data(path, data_name)
        self.transforms = transforms
        self.steps = steps

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x, y = self.x[None, index], self.y[None, index]
        x, y = x.astype('float32')[0] / 1., y.astype('float32')[0] / 1.

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y

    def load_data(self, path, data_name):
        #
        print('loading ' + data_name + ' data...')

        folders = os.listdir(path)
        imgs_list = []
        masks_list = []
        imgs_items = os.listdir(os.path.join(path, 'x'))
        for item in imgs_items:
            img_temp = np.load(os.path.join(path, 'x', item))
            imgs_list.append(img_temp)
            mask_temp = np.load(os.path.join(path, 'y', item))
            masks_list.append(mask_temp)

        imgs_np = np.asarray(imgs_list)
        masks_np = np.asarray(masks_list)

        x = np.asarray(imgs_np)
        y = np.asarray(masks_np)
        # x为data，y为label
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        print("Successfully loaded data from " + path)
        print("data shape:", "x.shape", x.shape, "y.shape", y.shape)

        return x, y
