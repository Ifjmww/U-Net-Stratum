import os
import time
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import importlib


def argparsing():
    parser = argparse.ArgumentParser(description='U-Net')
    parser.add_argument('--epochs', default=100, type=int, help='trining epochs')
    parser.add_argument('--batch_size', default=5, type=int, help='batch size')
    parser.add_argument('--batch_size_pred', default=1, type=int, help='batch size')
    parser.add_argument('--steps', default=500, type=int, help='steps per epoch')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=3e-5, type=float, help='learning rate decay')
    parser.add_argument('--num_class', default=10, type=int, help='model output channel number')
    parser.add_argument('--save_weight', action='store_true', help='save weight only')
    parser.add_argument('--train_data', default='./data/train', type=str, help='data path')
    parser.add_argument('--valid_data', default='./data/test', type=str, help='data path')
    parser.add_argument('--exp', default='1', type=str, help='experiment number')
    parser.add_argument('--evaluate_only', action='store_true', help='evaluate only?')
    parser.add_argument('--save_result', action='store_true', default=True, help='save results to exp folder?')
    parser.add_argument('--model_path', default=None, type=str, help='path to model check')
    parser.add_argument('--valid_dataset', default='Stratum', choices=['Stratum'], type=str, help='which dataset to validate?')
    parser.add_argument('--backend', default='pytorch', choices=['keras', 'pytorch'], type=str, help='which backend to use?')
    parser.add_argument('--prediction_data', default='../data/prediction', type=str, help='data path')
    parser.add_argument('--prediction_only', action='store_true', help='prediction?')
    args = parser.parse_args()

    print()
    print("============================================================")
    print(args)
    print()
    print("============================================================")
    print()
    return args


def main(args, CORE):
    # 检查文件路径的合法性
    if args.model_path is not None:
        if os.path.isfile(args.model_path):
            print('Model path has been verified.')
        else:
            print('Invalid model path! Please specify a valid model file. Program terminating...')
            exit()

    if args.prediction_only:
        # 测试
        CORE.prediction(args)
        exit()
    if not args.evaluate_only:
        time_start = time.time()
        # 训练
        CORE.train(args)
        time_end = time.time()
        time_sum = time_end - time_start
        print("===========TIME==========", time_sum, "===========TIME==========")
    CORE.evaluate(args)
    print("====================================")


if __name__ == '__main__':
    args = argparsing()
    CORE = importlib.import_module(args.backend + '_version')
    main(args, CORE)
