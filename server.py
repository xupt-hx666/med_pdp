from attack import model_inversion_attack
from model import MedModel
from args import args_parser
from client import train, validate_personalization, validate
import torch.nn as nn
from crypto import PaillierEncryptor
# from cryp_2 import OU98Encryptor
import numpy as np
import torch

args = args_parser()


class FedPer:
    def __init__(self):
        self.args = args
        self.encryptor = PaillierEncryptor()
        # self.encryptor = OU98Encryptor()
        self.global_base = self.base_layers = nn.Sequential(
            nn.Linear(args.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(args.device)

        self.client_models = []
        for _ in range(args.K):
            model = MedModel(name=f"client_{_}").to(args.device)
            model.base_layers.load_state_dict(self.global_base.state_dict())
            self.client_models.append(model)

    def aggregate(self, encrypted_weights_list):
        """聚合加密参数"""
        averaged_weights = {}
        num = len(encrypted_weights_list)

        for key in encrypted_weights_list[0].keys():
            # 提取加密数据和形状
            encrypted_arrays = [w[key]["encrypted"] for w in encrypted_weights_list]
            shapes = [w[key]["shape"] for w in encrypted_weights_list]
            shape = encrypted_weights_list[0][key]["shape"]
            if not all(shape == shapes[0] for shape in shapes):
                raise ValueError("参数形状不一致")

            # 同态加法聚合
            # paillier
            summed = []
            for i in range(len(encrypted_arrays[0])):
                total = encrypted_arrays[0][i]
                for arr in encrypted_arrays[1:]:
                    total += arr[i]
                summed.append(total)

            # ou98
            # summed = []
            # for i in range(len(encrypted_arrays[0])):
            #     total = encrypted_arrays[0][i]
            #     for arr in encrypted_arrays[1:]:
            #         total = (total * arr[i]) % (self.encryptor.n**2)
            #     summed.append(total)

            # 解密并还原
            decrypted_tensor = self.encryptor.decrypt_tensor({
                "shape": shape,
                "encrypted": summed
            }).to(args.device)
            averaged_weights[key] = decrypted_tensor / num

        return averaged_weights

    def server_round(self, round_idx):
        num_selected = max(int(args.C * args.K), 1)
        """每一轮通信随机挑选客户端参与训练，这里是选择每轮全部客户端参与训练而没有随机挑选"""
        # selected_clients = np.random.choice(range(args.K), num_selected, replace=False)
        selected_clients = list(range(num_selected))

        for idx in selected_clients:
            self.client_models[idx].base_layers.load_state_dict(self.global_base.state_dict())

        encrypted_weights_list = []
        for idx in selected_clients:
            model = self.client_models[idx]
            model.base_layers.load_state_dict(self.global_base.state_dict())
            encrypted_weights = train(args, model, idx, self.encryptor)
            encrypted_weights_list.append(encrypted_weights)

        # 聚合模型
        averaged_weights = self.aggregate(encrypted_weights_list)
        self.global_base.load_state_dict(averaged_weights)

        val_accs = []
        for idx in selected_clients:
            acc = validate(args, self.client_models[idx], idx)
            val_accs.append(acc)
            print(f"Client {idx} Val Acc: {acc:.2f}%")
        print(f"Round Average Val Acc: {sum(val_accs) / len(val_accs):.2f}%")

    def get_personal_params(self):
        """获取所有客户端的个性化层参数"""
        personal_params = []
        for model in self.client_models:
            params = []
            for param in model.personal_layers.parameters():
                params.append(param.data.cpu().numpy())
            personal_params.append(np.concatenate([p.flatten() for p in params]))
        print(np.array(personal_params))

    def run(self):
        for r in range(args.r):
            print(f"\n=== Round {r + 1}/{args.r} ===")
            self.server_round(r)

            print("\n=== 启动模型反演攻击测试 ===")
            target_client_id = 0
            target_label = 1
            fake_image = model_inversion_attack(
                self.client_models[target_client_id],
                target_label,
                self.args,
                attack_epochs=200
            )
        print("个性层差异")
        print("=======================")
        validate_personalization(self.client_models)

        torch.save(self.client_models[0].state_dict(), 'pneumonia_model.pth')
        print("模型已保存为 pneumonia_model.pth")
