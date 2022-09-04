from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torchsummary import summary
from .metrics import *
from .dataloader import UNetDataset
from .u_net_model import UNet
from .tools import *

device = torch.device('cuda:0')


def train(args):
    train_set = UNetDataset(args.train_data, 'Stratum', batchsize=args.batch_size, transforms=None)
    test_set = UNetDataset(args.valid_data, args.valid_dataset)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True,
                             pin_memory=True)

    # create model
    # model = UNet(iterations=args.iter, num_classes=args.num_class, num_layers=4, multiplier=args.multiplier, integrate=args.integrate).to(device).float()
    model = UNet(n_channels=1, n_classes=args.num_class).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=0.)

    fcn = lambda step: 1. / (1. + args.lr_decay * step)
    scheduler = LambdaLR(optimizer, lr_lambda=fcn)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    print('model successfully built and compiled.')

    if not os.path.isdir("checkpoints/" + args.exp):
        os.mkdir("checkpoints/" + args.exp)

    best_iou = 0.
    print('\nStart training...')

    for epoch in range(args.epochs):
        tot_loss = 0.
        tot_iou = 0.

        val_loss = 0.
        val_iou = 0.

        # training
        model.train()
        for step, (x, y) in enumerate(
                tqdm(train_loader, desc='[TRAIN] Epoch ' + str(epoch + 1) + '/' + str(args.epochs))):
            x = x.to(device).float()
            y = y.to(device).float()

            optimizer.zero_grad()
            output = model(x)
            # loss
            l = criterion(output, y.long())
            tot_loss += l.item()

            l.backward()

            optimizer.step()

            # metrics，x指的是预测值，y指的是真实值，.detach()是切断反向传播、去除梯度
            x, y = output.detach().cpu().numpy(), y.detach().cpu().numpy()
            iou_score = miou(y, x, args.num_class)

            tot_iou += iou_score
            # 学习率更新
        scheduler.step()

        print('[TRAIN] Epoch: ' + str(epoch + 1) + '/' + str(args.epochs),
              'loss:', tot_loss / len(train_loader),
              'iou:', tot_iou / len(train_loader))

        if not os.path.exists("checkpoints/" + args.exp + "/outputs"):
            os.mkdir("checkpoints/" + args.exp + "/outputs")

        with open("checkpoints/" + args.exp + "/outputs/train_result.txt", 'a+') as f:
            f.write('epoch:\t' + str(epoch) + '\n')
            f.write('Train  loss:\t' + str(tot_loss / len(train_loader)) + '\n')
            f.write('Train  iou:\t' + str(tot_iou / len(train_loader)) + '\n')

        # validation
        model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(
                    tqdm(test_loader, desc='[VAL] Epoch ' + str(epoch + 1) + '/' + str(args.epochs))):
                x = x.to(device).float()
                y = y.to(device).float()

                output = model(x)

                # loss
                l = criterion(output, y.long())
                val_loss += l.item()

                # metrics
                x, y = output.detach().cpu().numpy(), y.cpu().numpy()
                iou_score = miou(y, x, args.num_class)
                val_iou += iou_score

        if val_iou / len(test_loader) >= best_iou:
            best_iou = val_iou / len(test_loader)
            save_model(args, model)
            # 存下来的是最好的模型

        print('[VAL] Epoch: ' + str(epoch + 1) + '/' + str(args.epochs),
              'val_loss:', val_loss / len(test_loader),
              'val_iou:', val_iou / len(test_loader),
              'best val_iou:', best_iou)

        if not os.path.exists("checkpoints/" + args.exp + "/outputs"):
            os.mkdir("checkpoints/" + args.exp + "/outputs")

        with open("checkpoints/" + args.exp + "/outputs/eval_result.txt", 'a+') as f:
            f.write('epoch:\t' + str(epoch) + '\n')
            f.write('Validation loss:\t' + str(val_loss / len(test_loader)) + '\n')
            f.write('Validation  iou:\t' + str(val_iou / len(test_loader)) + '\n')

    print('\nTraining finished!')


def evaluate(args):
    test_set = UNetDataset(args.valid_data, args.valid_dataset)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True,
                             pin_memory=True)

    if args.model_path is None:
        integrate = '_int' if args.integrate else ''
        weights = '_weights'
        cpt_name = 'iter_' + str(args.iter) + '_mul_' + str(args.multiplier) + integrate + '_best' + weights + '.pt'
        model_path = "checkpoints/" + args.exp + "/" + cpt_name
    else:
        model_path = args.model_path
    print('Restoring model from path: ' + model_path)
    # model = UNet(iterations=args.iter, num_classes=args.num_class, num_layers=4, multiplier=args.multiplier,
    #              integrate=args.integrate).to(device)
    model = UNet(n_channels=1, n_classes=args.num_class).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    # 加载参数
    criterion = CrossEntropyLoss()

    val_loss = 0.
    val_iou = 0.

    segmentations = []
    inputs = []
    masks = []
    # validation
    print('\nStart evaluation...')
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(tqdm(test_loader)):

            x = x.to(device).float()
            y = y.to(device).float()

            output = model(x)

            # loss
            l = criterion(output, y.long())
            val_loss += l.item()

            # metrics
            input = x.detach().cpu().numpy()
            x, y = output.detach().cpu().numpy(), y.cpu().numpy()
            iou_score = miou(y, x, args.num_class)
            val_iou += iou_score

            # x代表的是output
            if args.save_result:
                segmentations.append(x)
                inputs.append(input)
                masks.append(y)

    val_loss = val_loss / len(test_loader)
    val_iou = val_iou / len(test_loader)

    print('Validation loss:\t', val_loss)
    print('Validation  iou:\t', val_iou)

    # summary(model, input_size=(1, 256, 256))
    macs, params = get_flops(args, model)
    print('\nEvaluation finished!')

    if args.save_result:
        if not os.path.exists("checkpoints/" + args.exp + "/outputs"):
            os.mkdir("checkpoints/" + args.exp + "/outputs")
        if not os.path.exists("checkpoints/" + args.exp + "/outputs/segmentations"):
            os.mkdir("checkpoints/" + args.exp + "/outputs/segmentations")
        with open("checkpoints/" + args.exp + "/outputs/result.txt", 'w+') as f:
            f.write('Validation loss:\t' + str(val_loss) + '\n')
            f.write('Validation  iou:\t' + str(val_iou) + '\n')
            f.write('macs:\t' + str(macs) + '\n')
            f.write('params:\t' + str(params) + '\n')

        print('Metrics have been saved to:', "checkpoints/" + args.exp + "/outputs/result.txt")

        print('Saving segmentations...')
        count = 0
        for i in range(len(segmentations)):
            for batch in range(args.batch_size):
                np.save("checkpoints/" + args.exp + "/outputs/segmentations/" + str(count) + ".npy",
                        segmentations[i][batch, :, :, :])
                np.save("checkpoints/" + args.exp + "/outputs/segmentations/" + str(count) + "_x.npy",
                        inputs[i][batch, :, :, :])
                np.save("checkpoints/" + args.exp + "/outputs/segmentations/" + str(count) + "_y.npy",
                        masks[i][batch, :, :])
                count += 1

        print("segmentation results have been saved!!!")
        # display函数，保存结果&结果可视化
        display(args, len(test_loader))
        print("===========================================")
        print("display have been finished!!!")


def prediction(args):
    pred_set = UNetDataset(args.prediction_data, args.pred_dataset)
    pred_loader = DataLoader(dataset=pred_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True,
                             pin_memory=True)

    if args.model_path is None:
        integrate = '_int' if args.integrate else ''
        weights = '_weights'
        cpt_name = 'iter_' + str(args.iter) + '_mul_' + str(args.multiplier) + integrate + '_best' + weights + '.pt'
        model_path = "checkpoints/" + args.exp + "/" + cpt_name
    else:
        model_path = args.model_path

    print('Restoring model from path: ' + model_path)
    # model = UNet(iterations=args.iter, num_classes=args.num_class, num_layers=4, multiplier=args.multiplier,
    #              integrate=args.integrate).to(device)
    model = UNet(n_channels=1, n_classes=args.num_class).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    # 加载参数
    segmentations = []
    # validation
    print('\nStart evaluation...')
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(tqdm(pred_loader)):

            x = x.to(device).float()
            output = model(x)
            # metrics
            x = output.detach().cpu().numpy()

            # x代表的是output
            if args.save_result:
                segmentations.append(x)

    print('\nEvaluation finished!')

    if args.save_result:

        # save metrics 新建output文件夹，将val_loss,val_iou,val_dice写入result.tx
        # save segmentations 保存输出的图像
        # results = np.transpose(np.concatenate(segmentations, axis=0), (0, 2, 3, 1))
        # results = (results > 0.5).astype(np.float32)  # Binarization. Comment out this line if you don't want to

        print('Saving segmentations...')
        if not os.path.exists(args.prediction_data + "pred_result"):
            os.mkdir(args.prediction_data + "pred_result")
        count = 0
        for i in range(len(segmentations)):
            #
            for batch in range(args.batch_size):
                segment_temp = segmentations[i][batch, :, :, :]
                segment_temp = segment_temp.argmax(axis=0)
                segment_temp = np.reshape(segment_temp, (256, 256))
                np.save(args.prediction_data + "pred_result/" + str(count + 100) + ".npy", segment_temp)

                count += 1

        print("segmentation results have been saved!!!")
    splicing(args)

# =====================================================================================================================
