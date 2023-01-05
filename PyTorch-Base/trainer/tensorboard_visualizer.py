# # <font style="color:blue">TensorBoard Visualizer Class</font>

from torch.utils.tensorboard import SummaryWriter

from .visualizer import Visualizer

import torch
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TensorBoardVisualizer(Visualizer):
    def __init__(self):
        self._writer = SummaryWriter()


    def create_confusion_matrix(self, net, loader):
        y_pred = [] # save prediction
        y_true = [] # save ground truth

        # iterate over data
        for inputs, labels in loader:
            inputs = inputs.to('cuda')
            output = net(inputs)

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # save prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # save ground truth

        # constant for classes
        classes = (
            0,
            30,
            60,
            90,
        )

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
        df_cm = pd.DataFrame(
            cf_matrix, index=[i for i in classes],
            columns=[i for i in classes]
        )
        plt.figure(figsize=(12, 7))
        return sn.heatmap(df_cm, annot=True).get_figure()

    def update_charts(self, train_metric, train_loss, test_metric, test_loss, learning_rate, epoch, model, loader):
        if train_metric is not None:
            for metric_key, metric_value in train_metric.items():
                self._writer.add_scalar("data/train_metric:{}".format(metric_key), metric_value, epoch)

        for test_metric_key, test_metric_value in test_metric.items():
            self._writer.add_scalar("data/test_metric:{}".format(test_metric_key), test_metric_value, epoch)

        if train_loss is not None:
            self._writer.add_scalar("data/train_loss", train_loss, epoch)
        if test_loss is not None:
            self._writer.add_scalar("data/test_loss", test_loss, epoch)

        self._writer.add_scalar("data/learning_rate", learning_rate, epoch)

        self._writer.add_figure("Test Confusion Matrix", self.create_confusion_matrix(model, loader))

    def close_tensorboard(self):
        self._writer.close()
