import numpy as np
from sklearn.metrics import confusion_matrix


def miou(y_true, y_pred, num_class=10):
    """
    compute mean IoU
    """
    y_pred = y_pred.argmax(axis=1).flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=range(num_class))
    intersection = np.diag(current)

    ground_truth_set = current.sum(axis=1)
    # 按行求和
    predicted_set = current.sum(axis=0)
    # 按列求和
    union = ground_truth_set + predicted_set - intersection + 1e-7
    IoU = intersection / union.astype(np.float32)

    return np.mean(IoU)
    # 返回十个类别的平均iou
