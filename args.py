import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--E', type=int, default=20, help='客户端本地训练轮数')
    parser.add_argument('--r', type=int, default=3, help='通信轮数')
    parser.add_argument('--K', type=int, default=3, help='客户端总数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--img_size', type=int, default=28, help='输入图像尺寸')
    parser.add_argument('--num_classes', type=int, default=2, help='肺炎二分类')
    parser.add_argument('--input_dim', type=int, default=784, help='输入特征维度（例如28x28=784）')
    parser.add_argument('--C', type=float, default=1.0, help='每轮参与客户端比例')
    parser.add_argument('--B', type=int, default=32, help='本地批量大小')
    parser.add_argument('--optimizer', type=str, default='adam', help='优化器')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--sigma", type=float, default=0.01, help="用于控制高斯噪声强度")
    parser.add_argument("--max_grad_threshold", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument("--epsilon", type=float, default=1.0, help="差分隐私预算")
    parser.add_argument("--delta", type=float, default=1e-5, help="δ值")

    args = parser.parse_args()
    return args

