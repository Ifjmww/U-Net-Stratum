import sys
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def save_model(args, model):
    integrate = '_int' if args.integrate else ''
    weights = '_weights'
    cpt_name = 'iter_' + str(args.iter) + '_mul_' + str(args.multiplier) + integrate + '_best' + weights + '.pt'
    torch.save({'state_dict': model.state_dict()}, "checkpoints/" + args.exp + "/" + cpt_name)


def get_flops(args, model):
    """
    compute computations: MACs, #Params
    """

    from ptflops import get_model_complexity_info
    save_stdout = sys.stdout
    sys.stdout = open('./trash', 'w')
    macs, params = get_model_complexity_info(model, (1, 256, 256), as_strings=False, print_per_layer_stat=False, verbose=False)
    sys.stdout = save_stdout
    print("macs: %.4f x 10^9, num params: %.4f x 10^6" % (float(macs) * 1e-9, float(params) * 1e-6))
    return macs, params


def display(args, length):
    path = "checkpoints/" + args.exp + "/outputs/segmentations/"
    if not os.path.exists("checkpoints/" + args.exp + "/outputs/images/"):
        os.mkdir("checkpoints/" + args.exp + "/outputs/images/")
    savePath = "checkpoints/" + args.exp + "/outputs/images/"
    test_dataset_length = args.batch_size * length

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


def splicing(filename, types, args):
    # 将预测出的图片拼在一起
    print('Splicing segmentations...')
    if not os.path.exists('./checkpoints/' + args.exp + '/pred/'):
        os.mkdir('./checkpoints/' + args.exp + '/pred/')

    crossline_crop_size = (58, 25, 709, 487)
    inline_crop_size = (36, 25, 987, 487)
    file_path = './checkpoints/' + args.exp + '/pred/' + filename + '/seg/'
    save_path = './checkpoints/' + args.exp + '/pred/'
    if types == 'crossline':
        dir_list = os.listdir(file_path)
        print(dir_list)

        total = []
        for item in dir_list:
            temp = np.load(os.path.join(file_path, item))
            # print(temp)
            total.append(temp)
        x_axis_0 = total[0]
        x_axis_1 = total[2]
        x_axis_2 = total[4]
        x_axis_3 = total[10]
        x_axis_4 = total[12]
        x_axis_5 = total[14]
        x_cat_0 = np.concatenate((x_axis_0, x_axis_1, x_axis_2), axis=1)
        x_cat_1 = np.concatenate((x_axis_3, x_axis_4, x_axis_5), axis=1)
        x_cat = np.concatenate((x_cat_0, x_cat_1), axis=0)

        for i in range(0, 206):
            for j in range(206, 306):
                x_cat[i, j] = total[1][i, j - 128]
            for k in range(462, 562):
                x_cat[i, k] = total[3][i, k - 384]

        for i in range(206, 306):
            for j in range(0, 206):
                x_cat[i, j] = total[5][i - 128, j]
            for k in range(306, 462):
                x_cat[i, k] = total[7][i - 128, k - 256]
            for m in range(562, 768):
                x_cat[i, m] = total[9][i - 128, m - 512]
            for center_1 in range(206, 306):
                x_cat[i, center_1] = total[6][i - 128, center_1 - 128]
            for center_2 in range(462, 562):
                x_cat[i, center_2] = total[8][i - 128, center_2 - 384]

        for i in range(306, 512):
            for j in range(206, 306):
                x_cat[i, j] = total[11][i - 256, j - 128]
            for k in range(462, 562):
                x_cat[i, k] = total[13][i - 256, k - 384]

        x_img = Image.fromarray(np.uint8(x_cat))
        x_crop = x_img.crop(crossline_crop_size)
        x_cat = np.array(x_crop)
        np.save(save_path + filename + ".npy", x_cat)
        plt.imshow(x_cat)
        plt.savefig(save_path + filename + ".png")

    else:
        dir_list = os.listdir(file_path)
        print(dir_list)

        total = []
        for item in dir_list:
            temp = np.load(os.path.join(file_path, item))
            # print(temp)
            total.append(temp)
        x_axis_0 = total[0]
        x_axis_1 = total[2]
        x_axis_2 = total[4]
        x_axis_3 = total[6]
        x_axis_4 = total[14]
        x_axis_5 = total[16]
        x_axis_6 = total[18]
        x_axis_7 = total[20]
        x_cat_0 = np.concatenate((x_axis_0, x_axis_1, x_axis_2, x_axis_3), axis=1)
        x_cat_1 = np.concatenate((x_axis_4, x_axis_5, x_axis_6, x_axis_7), axis=1)
        x_cat = np.concatenate((x_cat_0, x_cat_1), axis=0)

        for i in range(0, 206):
            for j in range(206, 306):
                x_cat[i, j] = total[1][i, j - 128]
            for k in range(462, 562):
                x_cat[i, k] = total[3][i, k - 384]
            for m in range(718, 818):
                x_cat[i, m] = total[5][i, m - 640]

        for i in range(206, 306):
            for j in range(0, 206):
                x_cat[i, j] = total[7][i - 128, j]
            for k in range(306, 462):
                x_cat[i, k] = total[9][i - 128, k - 256]
            for m in range(562, 718):
                x_cat[i, m] = total[11][i - 128, m - 512]
            for n in range(818, 1024):
                x_cat[i, n] = total[13][i - 128, n - 768]

            for center_1 in range(206, 306):
                x_cat[i, center_1] = total[8][i - 128, center_1 - 128]
            for center_2 in range(462, 562):
                x_cat[i, center_2] = total[10][i - 128, center_2 - 384]
            for center_3 in range(718, 818):
                x_cat[i, center_3] = total[12][i - 128, center_3 - 640]

        for i in range(306, 512):
            for j in range(206, 306):
                x_cat[i, j] = total[15][i - 256, j - 128]
            for k in range(462, 562):
                x_cat[i, k] = total[17][i - 256, k - 384]
            for m in range(718, 818):
                x_cat[i, m] = total[19][i - 256, m - 640]

        x_img = Image.fromarray(np.uint8(x_cat))
        x_crop = x_img.crop(inline_crop_size)
        x_cat = np.array(x_crop)
        np.save(save_path + filename + ".npy", x_cat)
        plt.imshow(x_cat)
        plt.savefig(save_path + filename + ".png")

    print("Splicing finished!!!")
    return x_cat
