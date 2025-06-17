from Crypto.Util.number import getPrime, inverse
import numpy as np
from tqdm import tqdm
import torch
from args import args_parser
import random

args = args_parser()


class OU98Encryptor:
    def __init__(self, key_size=64):
        p = getPrime(key_size)
        q = getPrime(key_size)
        self.n = p * q
        self.g = self.n + 1
        self.public_key = (self.n, self.g)
        self.private_key = (p, q)
        self.lambda_val = (p - 1) * (q - 1)

    def r_func(self):
        while True:
            r = random.randint(1, self.n - 1)
            if np.gcd(r, self.n) == 1:
                return r

    def encrypt_tensor(self, tensor, progress=True):
        scaled = (tensor.cpu().numpy() * 1e4 + 1e4).astype(int)
        encrypted = []
        with tqdm(total=scaled.size, desc="Encrypting", disable=not progress) as pbar:
            for x in scaled.flatten():
                x = int(x)
                r = self.r_func()
                c = (pow(self.g, x, self.n ** 2) * pow(r, self.n, self.n ** 2)) % self.n ** 2
                encrypted.append(c)
                pbar.update(1)
        return {"shape": tensor.shape, "encrypted": encrypted}

    def decrypt_tensor(self, encrypted_data, progress=True):
        p, q = self.private_key
        n = self.n
        decrypted = []
        with tqdm(total=len(encrypted_data["encrypted"]), desc="Decrypting", disable=not progress) as pbar:
            g_lambda = pow(self.g, self.lambda_val, n ** 2)
            L_g = (g_lambda - 1) // n
            mu_inv = inverse(L_g % n, n)

            for c in encrypted_data["encrypted"]:
                c = int(c)
                cp = pow(c, self.lambda_val, n ** 2)
                L = (cp - 1) // n
                m = (L * mu_inv) % n
                decrypted.append(float(m))
                pbar.update(1)
        decrypted_np = (np.array(decrypted).reshape(encrypted_data["shape"]) - 1e4 * args.K * args.C) / 1e4
        return torch.from_numpy(decrypted_np).float()
