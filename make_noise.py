import os
import numpy as np
from matplotlib import pyplot as plt


def get_noise_model_fix(data, noise_stype):
    tokens = noise_stype.split(',')
    stddev = float(tokens[1])
    noise = np.random.rand(*data.shape) * (stddev)
    data_noise = data + noise
    return data_noise


def process(file_path, noise_stype, save_path):
    files = os.listdir(file_path)
    print(files)
    data_shape = np.load(os.path.join(file_path, files[0])).shape

    seismic3d = np.zeros(shape=(len(files), data_shape[0], data_shape[1]))
    count = 0
    for item in files:
        data = np.load(os.path.join(file_path, item))
        filename = item.split('.')[0]
        data_noise = get_noise_model_fix(data, noise_stype)
        seismic3d[count] = data_noise
        print(count)
        count += 1
        np.save(os.path.join(save_path, filename) + '.npy', data_noise)
    seismic3d_mean = seismic3d.mean()
    seismic3d_std = seismic3d.std()
    new_files = os.listdir(save_path)
    for item in new_files:
        temp = np.load(os.path.join(save_path, item))
        filename = item.split('.')[0]
        temp = (temp - seismic3d_mean) / seismic3d_std
        np.save(os.path.join(save_path, filename) + '.npy', temp)


def display(file_path, save_path, label_path):
    data_list = os.listdir(file_path)
    noise_list = os.listdir(save_path)
    label_list = os.listdir(label_path)
    data = np.load(os.path.join(file_path, data_list[0]))
    noisa_data = np.load(os.path.join(save_path, noise_list[0]))
    label = np.load(os.path.join(label_path, label_list[0]))

    plt.figure(figsize=(50, 50))
    plt.subplot(1, 3, 1)
    plt.imshow(data)
    plt.title("data")

    plt.subplot(1, 3, 2)
    plt.imshow(noisa_data)
    plt.title("label")

    plt.subplot(1, 3, 3)
    plt.imshow(label)
    plt.title("segmentation")
    plt.show()


# file_path = './data/train/x/'
# label_path = './dataWithNoise/10/train/y/'
# save_path = './dataWithNoise/30/train/x/'
file_path = './data/val/x/'
label_path = './dataWithNoise/10/val/y/'
save_path = './dataWithNoise/30/val/x/'


noise_stype = 'guassion,0.30'

process(file_path, noise_stype, save_path)
display(file_path, save_path, label_path)