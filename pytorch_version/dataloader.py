from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
from PIL import Image
import numpy as np
import torch
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt


class UNetDataset(Dataset):
    # load all data into cpu, consistent to keras version.与Keras版本相同，把所有数据传入cpu中
    def __init__(self, path, data_name, batchsize=1, steps=None, shuffle=False, transforms=None):
        self.x, self.y = self.load_data(path, data_name)
        self.transforms = transforms
        self.steps = steps
        if steps is not None:
            self.idx_mapping = np.random.randint(0, self.x.shape[0], steps * batchsize)
            # 输出大小为steps*batchsize的，范围为0到x.shape[0]的随机数组 # 原始数据序列的一个扩充序列
            self.steps = self.steps * batchsize

    def __len__(self):
        # 返回数据集的大小
        return self.steps if self.steps is not None else self.x.shape[0]
        # 如果steps为None，返回x.shape[0],如果steps不为None，返回steps

    def __getitem__(self, index):
        # 按索引取值
        if self.steps is not None:
            index = self.idx_mapping[index]

        x, y = self.x[None, index], self.y[None, index]

        x, y = x.astype('float64')[0] / 1., y.astype('float64')[0] / 1.
        # x, y = ToTensor()(x), ToTensor()(y)
        #
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y

    def load_data(self, path, data_name):
        #
        print('loading ' + data_name + ' data...')

        folders = os.listdir(path)
        # 打开文件夹
        imgs_list = []
        masks_list = []
        # 设置两个空列表，一个存data，一个存label
        imgs_items = os.listdir(os.path.join(path, 'x'))
        masks_items = os.listdir(os.path.join(path, 'y'))
        # x文件夹里面全是data，y文件夹里面全是label
        for item in imgs_items:
            img_temp = np.load(os.path.join(path, 'x', item))
            imgs_list.append(img_temp)
            mask_temp = np.load(os.path.join(path, 'y', item))
            masks_list.append(mask_temp)
            # print(item)

        imgs_np = np.asarray(imgs_list)
        masks_np = np.asarray(masks_list)
        # print('imgs_np.shape',imgs_np.shape,'masks_np.shape',masks_np.shape)

        x = np.asarray(imgs_np)
        y = np.asarray(masks_np)
        # x为data，y为label
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
            # 如果标签的维度是3，给变为4维

        print("Successfully loaded data from " + path)
        print("data shape:", "x.shape", x.shape, "y.shape", y.shape)

        return x, y
