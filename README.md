# U-Net-Stratum
基于简化的U-Net网络模型的荷兰F3地震层序识别


## 环境
- python 3.7
- pytorch 1.9.1
- CUDA 11.1

## 训练及验证

```commandline
python main.py --epochs [epoch_number] --train_data [train_data_path] --valid_data [val_data_path] --exp [experiment_name]
```
(仅给出部分参数，详细参数设置详见 main.py)

## 预测

```commandline
python main.py --prediction_only --prediction_data [pred_data_path] --exp [experiment_name]
```

## 结果展示

![Crossline](./images/Crossline.jpg "Crossline")

![Inline](./images/Inline.jpg "Inline")


