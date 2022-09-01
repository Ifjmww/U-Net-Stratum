import numpy as np
import os
import matplotlib.pyplot as plt

path = '../checkpoints/iter_4_data/outputs/segmentations/'
savePath = '../checkpoints/iter_4_data/outputs/images/'
test_dataset_length = 80

for len in range(test_dataset_length):
    seg_name = str(len) + '.npy'
    name = seg_name.split('.')
    data_name = name[0] + '_x.npy'
    label_name = name[0] + '_y.npy'

    seg_path = os.path.join(path, seg_name)
    x_path = os.path.join(path, data_name)
    y_path = os.path.join(path, label_name)

    seg = np.load(seg_path)
    x = np.load(x_path)
    x = np.reshape(x, (x.shape[1], x.shape[2]))
    y = np.load(y_path)

    # output = np.zeros((seg.shape[1], seg.shape[2]))
    #
    # for i in range(seg.shape[1]):
    #     for j in range(seg.shape[2]):
    #         maxItem = seg[0, i, j]
    #         max_index = 0
    #         for k in range(seg.shape[0]):
    #             if seg[k, i, j] > maxItem:
    #                 max_index = k
    #         output[i, j] = max_index
    output = seg.argmax(axis=0)

    plt.figure(figsize=(50, 50))
    plt.subplot(1, 3, 1)
    plt.imshow(x)
    plt.title("data")

    plt.subplot(1, 3, 2)
    plt.imshow(y)
    plt.title("label")

    plt.subplot(1, 3, 3)
    plt.imshow(output)
    plt.title("segmentation")

    save_Path = os.path.join(savePath, str(len) + '.png')
    print(save_Path)
    plt.savefig(save_Path)
    # plt.show()
