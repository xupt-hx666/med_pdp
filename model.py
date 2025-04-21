import torch.nn as nn
from args import args_parser

args = args_parser()


class MedModel(nn.Module):
    def __init__(self, name):
        super(MedModel, self).__init__()
        self.name = name

        self.base_layers = nn.Sequential(
            nn.Linear(args.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.personal_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, args.num_classes)
        )

    def forward(self, x):
        x = x.view(-1, args.input_dim)  # 展平输入
        x = self.base_layers(x)
        return self.personal_layers(x)
