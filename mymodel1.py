#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Model definitions used by the ADS-B classification training script.

Only the :class:`CNN_Transformer` architecture is retained because it offers
the best trade-off between capacity and generalisation on the 10-class ADS-B
dataset.  Earlier revisions bundled multiple legacy CNN variants that were no
longer exercised anywhere in the code base; removing them reduces the
maintenance surface and keeps the modelling story unambiguous.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CNN_Transformer"]


class CNN_Transformer(nn.Module):
    """Compact CNN + Transformer network tailored for 10-way ADS-B recognition.

    The architecture intentionally keeps the channel counts and Transformer
    dimensionality modest (ending at 192 hidden units) to match the 10-class
    objective and reduce overfitting risk.  Convolutions learn local temporal
    patterns, while the Transformer encoder captures long-range dependencies.

    Parameters
    ----------
    num_cls:
        Number of output classes.  Defaults to 10 for the ADS-B dataset but can
        be overridden for other experiments.
    """

    def __init__(self, num_cls: int = 10) -> None:
        super().__init__()

        # The convolutional trunk uses modest channel counts and GELU
        # activations.  Pooling every block halves the temporal resolution so
        # the subsequent Transformer processes a shorter sequence.
        self.features = nn.Sequential(
            nn.Conv1d(2, 24, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(24),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(24, 48, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(48, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(96, 144, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(144),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(144, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=3,
            dim_feedforward=384,
            dropout=0.2,
            activation="gelu",
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            norm=nn.LayerNorm(192),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Dropout(0.25),
            nn.Linear(192, num_cls),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x:
            Batch of complex I/Q sequences with shape ``(batch, 2, 4800)`` where
            the second dimension enumerates the I and Q components and the last
            dimension is the temporal axis.

        Returns
        -------
        logits:
            Raw, unnormalised class scores with shape ``(batch, num_cls)``.
        features:
            The pooled feature representation prior to the classifier.  This is
            handy for downstream analysis such as t-SNE visualisation.
        """

        # Convolutional feature extractor operating on channel-first 1-D data.
        x = self.features(x)

        # Prepare for the Transformer encoder: convert to (sequence, batch, feature).
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)

        # Restore (batch, feature, sequence) and compute global average pooling
        # over the temporal dimension instead of taking only the final token.
        x = x.permute(1, 2, 0)
        features = x.mean(dim=2)

        logits = self.classifier(features)
        return logits, features


# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
# import x_transformers as xt
from gconv import GConv2d

# 通道注意力机制


class ConvNet_5G3_3(nn.Module):# 改进了一下ConvNet_5G2 输入选择为6000点
    def __init__(self, num_cls=10):
        super(ConvNet_5G3_3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 2 * 25, kernel_size=15, stride=1, padding=7, bias=True),
            nn.BatchNorm1d(2 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(2 * 25, 3 * 25, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm1d(3 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),

            nn.Conv1d(3 * 25, 4 * 25, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(4 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(4 * 25, 6 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(6 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(6 * 25, 8 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(8 * 25),
            nn.ReLU(inplace=True),
            nn.Conv1d(8 * 25, 8 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(8 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            # nn.Dropout(0.25),

            nn.Conv1d(8 * 25, 12 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.Conv1d(12 * 25, 12 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(12 * 25, 12 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.Conv1d(12 * 25, 12 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(12 * 25, 12 * 25, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
        )
        self.fc1 = nn.Linear(300 * 6, 300)  #
        self.prelu_fc1 = nn.PReLU()
        self.Dropout = nn.Dropout(0.6)
        self.fc3 = nn.Linear(300, num_cls)  # 没有进行归一化，转化为独热编码

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
            # print(x.shape)
        # print(x.shape)
        x = x.view(-1, 300 * 6)
        x = self.prelu_fc1(self.fc1(x))
        x = self.Dropout(x)
        feature = x
        y = self.fc3(x)
        return y , feature



class CNN_Transformer_memory(nn.Module):
    def __init__(self, num_cls=100):
        super(CNN_Transformer_memory, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 4 * 25, kernel_size=15, stride=1, padding=7, bias=True),
            nn.BatchNorm1d(4 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4),

            nn.Conv1d(4 * 25, 8 * 25, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm1d(8 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(8 * 25, 12 * 25, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(12 * 25, 16 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(16 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(16 * 25, 24 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(24 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(24 * 25, 32 * 25, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(32 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(32 * 25, 36 * 25, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(36 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(36 * 25, 1024, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
        )

        # self.encoder_layer = xt.Encoder(dim=512, depth = 6, heads = 8, attn_num_mem_kv = 16,  layer_dropout = 0.2, attn_dropout = 0.2, ff_dropout = 0.2)#
        # self.encoder_layer = xt.Encoder(  dim = 1024,#  输入特征维度
        #                                   depth = 3,#  层数
        #                                   heads = 8,#  头数
        #                                   attn_num_mem_kv = 256,   #  传入记忆向量参数
        #                                   layer_dropout=0.2,  #  dropout 整层
        #                                   attn_dropout=0.2,  # dropout post-attention
        #                                   ff_dropout=0.2  # 前馈 dropout
        #                                             )
        self.classifier1 = nn.Sequential(nn.PReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(1024, num_cls)
                                         )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        # print(x.size()) #   ([batchsize, d_fea, 68]) torch.Size([64, 512, 150])
        x = x.permute(2, 0, 1)
        # print(x.size()) # torch.Size([150, 64, 512])
        x = self.encoder_layer(x)
        # print(x.size())  # torch.Size([150, 64, 512])
        x = x.permute(1, 2, 0)
        # print(x.size()) # torch.Size([64, 512, 150])
        x = torch.mean(x, dim=2)
        # print(x.size())# torch.Size([64, 512])
        feature = x
        x = self.classifier1(x)
        return x ,feature


class CNN_Transformer(nn.Module):
    def __init__(self, num_cls=10):
        super(CNN_Transformer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 4 * 25, kernel_size=15, stride=1, padding=7, bias=True),
            nn.BatchNorm1d(4 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4),

            nn.Conv1d(4 * 25, 8 * 25, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm1d(8 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(8 * 25, 12 * 25, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(12 * 25, 16 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(16 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(16 * 25, 24 * 25, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(24 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(24 * 25, 32 * 25, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(32 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(32 * 25, 36 * 25, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(36 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(36 * 25, 1024, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
        )
        # 这个类是transformer encoder的组成部分，代表encoder的一个层，而encoder就是将transformerEncoderLayer重复几层。
        # 输入维度：seq, batch, feature
        # seqlenth x batch x dim 输入   多头注意力机制 10表示进行10次注意力运算之后通过线性运算转化为低维形式 需要d_model/nhead可以整除
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=1024*2, dropout = 0.2)#
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3, norm=nn.LayerNorm(1024))
        self.classifier1 = nn.Sequential(# nn.PReLU(),
                                         nn.Dropout(0.5),
                                         # nn.Linear(300, num_cls),
                                         nn.Linear(1024, num_cls)
                                         )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        # print(x.size()) #torch.Size([256, 512, 68])
        x = x.permute(2, 0, 1) # torch.Size([68, 256, 300])
        # x = torch.swapaxes (x,2, 0, 1)# x.swapaxes(2, 0, 1)
        # print(x.size()) # torch.Size([256, 68, 300])
        x = self.transformer_encoder(x)
        # print(x.shape)
        x = x.permute(1, 2, 0)
        x = x[:, :, -1]# 相当于这部分仅仅选取了最后的一个输出  改一下，采用全局平均值池化试试 这部分是不对的为什么仅仅取最后一个维度？？
        # x = torch.mean(x, dim=2)
        # print(x.size())# torch.Size([256, 300])
        feature = x
        x = self.classifier1(x)
        # AV = x  # 激活向量
        return x,feature


class P4AllCNN(nn.Module):
    """
    G-equivariant CNN with G = p4 (The 4 90-degree rotations)

    6 3x3 conv layers, followed by 4x4 conv layer (10 channels each layer)
    relu activation, bn, dropout, after layer 2
    max-pool after last layer
    """

    def __init__(self, n_classes=100):
        super().__init__()

        self.features = nn.Sequential(
            GConv2d(2, 32, filter_size=5, stride=1, padding=2),  # Reduced channels，2为输入通道数，32为输出通道数
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            # nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(128, 128, kernel_size=(1, 11), stride=1, padding=(0, 5), bias=True),#卷积核大小（1,11），说明主要捕捉宽度上的特征；bias（true）：卷积层中使用偏置项
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(1, 11), stride=1, padding=(0, 5), bias=True),#第二次卷积，和上面三行相同
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(1, 2)),#池化核尺寸，没说步幅，默认和池化核一样

            nn.Conv2d(128, 256, kernel_size=(1, 11), stride=1, padding=(0, 5), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(256, 512, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(512, 512, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(512, 1024, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(1024, 1024, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),#全连接层，线性层
            nn.Dropout(0.5),#正则化，随机丢弃50%的神经元，防止过拟合
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        feature = x
        # x = self.fc(x)
        for layer in self.classifier:
            x = layer(x)
        return x, feature


__factory = {

    'CNN':ConvNet_5G3_3,
    'CNN_Transformer':CNN_Transformer,
    'CNN_Transformer_memory': CNN_Transformer_memory,
    'P4AllCNN': P4AllCNN,
}

def create(name, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes)

