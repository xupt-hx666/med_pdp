import torch.nn as nn
from args import args_parser

args = args_parser()


class MedModel(nn.Module):
    def __init__(self, name):
        super(MedModel, self).__init__()
        self.name = name

        self.base_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 32, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 全局平均池化替代全连接
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()  # 展平特征图后输出维度为特征图通道数
        )

        self.personal_layers = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, args.num_classes)
        )

    def forward(self, x):  # 输入是 [batch,通道 ,h,w(h,w是图像尺寸，即高和宽)]
        x = self.base_layers(x)
        return self.personal_layers(x)
