import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset  # subset加载数据子集
from medmnist import PneumoniaMNIST
from args import args_parser

args = args_parser()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),  # RGB三通道，每一个通道代表一个颜色，图像灰度化则确定通道数为一
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = PneumoniaMNIST(split='train', transform=transform, download=True)
val_dataset = PneumoniaMNIST(split='val', transform=transform, download=True)
test_dataset = PneumoniaMNIST(split='test', transform=transform, download=True)


def load_data(client_id):
    # 均匀划分训练集给各客户端（IID划分）
    total_samples = len(train_dataset)
    samples_per_client = total_samples // args.K
    indices = list(range(client_id * samples_per_client, (client_id + 1) * samples_per_client))

    # 创建数据加载器
    train_loader = DataLoader(
        Subset(train_dataset, indices),
        batch_size=args.B,
        shuffle=True
    )

    val_loader = DataLoader(val_dataset, batch_size=args.B, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.B, shuffle=False)

    # print(inputs.shape in loader)  # [32, 1, 28, 28]
    # print(labels.shape)  # [B,1]数据加载器错误加载了维度，在后续标签处理时需要将标签压缩为一维

    return train_loader, val_loader, test_loader
