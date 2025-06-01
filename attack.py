"""模型反演攻击"""
# 根据模型输出推断输入数据
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image
import numpy as np
from medmnist import PneumoniaMNIST
from torchvision import transforms
import time


def model_inversion_attack(target_model, target_label, args, attack_epochs=100, lr=0.1):
    """模型反演攻击测试"""
    target_model.eval()

    device = args.device
    fake_input = torch.randn(1, args.input_dim, device=device).requires_grad_(True)

    original_mean = torch.tensor([0.5], device=device)
    original_std = torch.tensor([0.5], device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam([fake_input], lr=lr)

    test_dataset = PneumoniaMNIST(split='test', download=True, transform=None)
    real_samples = [img for img, label in test_dataset if label == target_label][:5]

    for epoch in range(attack_epochs):
        optimizer.zero_grad()
        processed_input = fake_input.view(1, 1, 28, 28)
        output = target_model(processed_input)
        target = torch.tensor([target_label], device=device)
        loss = criterion(output, target) + 0.01 * torch.norm(fake_input)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            time.sleep(2)
            print("-----反演攻击中-----")

    with torch.no_grad():
        fake_img = fake_input.view(1, 28, 28).cpu().detach()
        fake_img = fake_img * original_std.cpu() + original_mean.cpu()
        fake_img = torch.clamp(fake_img, 0, 1)
        save_image(fake_img, f'inversion_attack_label_{target_label}.png')

    similarity_scores = []
    for real_img in real_samples:
        real_tensor = transforms.ToTensor()(real_img).view(-1).to(device)
        fake_on_device = fake_input.flatten().to(device)
        score = torch.cosine_similarity(fake_on_device, real_tensor, dim=0)
        similarity_scores.append(score.item())

    print("----------反演攻击完成----------")
    print(f"[反演攻击] 目标标签 {target_label} 重建完成，与真实数据平均相似度: {np.mean(similarity_scores):.4f}")
    return fake_img
