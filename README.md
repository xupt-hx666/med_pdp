这个项目是在我发的上一个项目edu_pdp（基于联邦学习的个性化教育推荐）换了应用场景，本项目的应用场景为医疗。
两个项目的兼容度较高，项目的整体框架不变，做了以下改进来实现场景迁移：
1.换用医疗数据集medinist,图像的数据处理跟矩阵的处理方式不同，这里在数据处理时做了修改。
2.新增ou98加解密模块来实现除paillier以外的同态加密（根据想要使用的算法模块自己选择）

这里附上医疗场景的paillier模拟实验运行结果
(pytorch_gpu) C:\Users\coolboy\Desktop\deep_study\Medmodel_cry>python main.py
Using downloaded and verified file: C:\Users\coolboy\.medmnist\pneumoniamnist.npz
Using downloaded and verified file: C:\Users\coolboy\.medmnist\pneumoniamnist.npz
Using downloaded and verified file: C:\Users\coolboy\.medmnist\pneumoniamnist.npz

=== Round 1/3 ===
Client 0 Epoch 1/20 | Loss: 0.5749 | Acc: 73.80%
Client 0 Epoch 2/20 | Loss: 0.3497 | Acc: 82.41%
Client 0 Epoch 3/20 | Loss: 0.2063 | Acc: 93.44%
Client 0 Epoch 4/20 | Loss: 0.1591 | Acc: 93.37%
Client 0 Epoch 14/20 | Loss: 0.0821 | Acc: 96.75%
Client 0 Epoch 15/20 | Loss: 0.0885 | Acc: 96.88%
Client 0 Epoch 16/20 | Loss: 0.0755 | Acc: 97.32%
Client 0 Epoch 17/20 | Loss: 0.0724 | Acc: 97.32%
Client 0 Epoch 18/20 | Loss: 0.0612 | Acc: 97.64%
Client 0 Epoch 19/20 | Loss: 0.0681 | Acc: 97.77%
Client 0 Epoch 20/20 | Loss: 0.0640 | Acc: 97.51%
Encrypting: 100%|███████████████████████████████████████████████████████████████████████| 100352/100352 [10:26<00:00, 160.29it/s] 
Encrypting: 100%|█████████████████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 160.64it/s]
Client 1 Epoch 1/20 | Loss: 0.5576 | Acc: 73.23%
Client 1 Epoch 2/20 | Loss: 0.3138 | Acc: 84.07%
Client 1 Epoch 3/20 | Loss: 0.1867 | Acc: 93.56%
Client 1 Epoch 4/20 | Loss: 0.1472 | Acc: 95.03%
Client 1 Epoch 5/20 | Loss: 0.1346 | Acc: 95.41%
Client 1 Epoch 6/20 | Loss: 0.1264 | Acc: 95.54%
Client 1 Epoch 13/20 | Loss: 0.0849 | Acc: 97.32%
Client 1 Epoch 14/20 | Loss: 0.0724 | Acc: 97.71%
Client 1 Epoch 15/20 | Loss: 0.0674 | Acc: 97.58%
Client 1 Epoch 16/20 | Loss: 0.0673 | Acc: 97.58%
Client 1 Epoch 17/20 | Loss: 0.0555 | Acc: 97.83%
Client 1 Epoch 18/20 | Loss: 0.0654 | Acc: 97.32%
Client 1 Epoch 19/20 | Loss: 0.0558 | Acc: 97.83%
Client 1 Epoch 20/20 | Loss: 0.0471 | Acc: 98.66%
Encrypting: 100%|███████████████████████████████████████████████████████████████████████| 100352/100352 [10:20<00:00, 161.84it/s] 
Encrypting: 100%|█████████████████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 155.94it/s]
Client 2 Epoch 1/20 | Loss: 0.6008 | Acc: 70.81%
Client 2 Epoch 2/20 | Loss: 0.3443 | Acc: 83.56%
Client 2 Epoch 3/20 | Loss: 0.1939 | Acc: 92.67%
Client 2 Epoch 4/20 | Loss: 0.1486 | Acc: 94.46%
Client 2 Epoch 5/20 | Loss: 0.1260 | Acc: 95.22%
Client 2 Epoch 6/20 | Loss: 0.1170 | Acc: 95.41%
Client 2 Epoch 7/20 | Loss: 0.1127 | Acc: 95.79%
Client 2 Epoch 8/20 | Loss: 0.0950 | Acc: 95.98%
Client 2 Epoch 9/20 | Loss: 0.1001 | Acc: 95.98%
Client 2 Epoch 10/20 | Loss: 0.0867 | Acc: 97.20%
Client 2 Epoch 11/20 | Loss: 0.0973 | Acc: 96.43%
Client 2 Epoch 12/20 | Loss: 0.0876 | Acc: 97.13%
Client 2 Epoch 13/20 | Loss: 0.0695 | Acc: 97.13%
Client 2 Epoch 14/20 | Loss: 0.0716 | Acc: 97.39%
Client 2 Epoch 15/20 | Loss: 0.0584 | Acc: 98.15%
Client 2 Epoch 16/20 | Loss: 0.0569 | Acc: 98.15%
Client 2 Epoch 17/20 | Loss: 0.0621 | Acc: 97.45%
Client 2 Epoch 18/20 | Loss: 0.0646 | Acc: 98.34%
Client 2 Epoch 19/20 | Loss: 0.0580 | Acc: 97.90%
Client 2 Epoch 20/20 | Loss: 0.0417 | Acc: 98.53%
Encrypting: 100%|███████████████████████████████████████████████████████████████████████| 100352/100352 [10:35<00:00, 157.96it/s]
Encrypting: 100%|█████████████████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 159.97it/s] 
Decrypting: 100%|███████████████████████████████████████████████████████████████████████| 100352/100352 [02:53<00:00, 577.95it/s]
Decrypting: 100%|█████████████████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 543.38it/s]
Client 0 Val Acc: 95.23%
Client 1 Val Acc: 95.04%
Client 2 Val Acc: 95.80%
Round Average Val Acc: 95.36%

=== Round 2/3 ===
Client 0 Epoch 1/20 | Loss: 0.1080 | Acc: 95.98%
Client 0 Epoch 2/20 | Loss: 0.0987 | Acc: 96.43%
Client 0 Epoch 3/20 | Loss: 0.0927 | Acc: 96.62%
Client 0 Epoch 4/20 | Loss: 0.0792 | Acc: 97.32%
Client 0 Epoch 5/20 | Loss: 0.0825 | Acc: 97.32%
Client 0 Epoch 6/20 | Loss: 0.0750 | Acc: 97.51%
Client 0 Epoch 7/20 | Loss: 0.0612 | Acc: 97.90%
Client 0 Epoch 8/20 | Loss: 0.0657 | Acc: 97.51%
Client 0 Epoch 9/20 | Loss: 0.0583 | Acc: 97.64%
Client 0 Epoch 10/20 | Loss: 0.0605 | Acc: 98.28%
Client 0 Epoch 11/20 | Loss: 0.0532 | Acc: 98.09%
Client 0 Epoch 12/20 | Loss: 0.0530 | Acc: 98.09%
Client 0 Epoch 13/20 | Loss: 0.0481 | Acc: 98.15%
Client 0 Epoch 14/20 | Loss: 0.0409 | Acc: 98.73%
Client 0 Epoch 15/20 | Loss: 0.0416 | Acc: 98.66%
Client 0 Epoch 16/20 | Loss: 0.0448 | Acc: 98.15%
Client 0 Epoch 17/20 | Loss: 0.0494 | Acc: 98.47%
Client 0 Epoch 18/20 | Loss: 0.0409 | Acc: 99.11%
Client 0 Epoch 19/20 | Loss: 0.0455 | Acc: 97.96%
Client 0 Epoch 20/20 | Loss: 0.0414 | Acc: 98.53%
Encrypting: 100%|███████████████████████████████████████████████████████████████████████| 100352/100352 [09:46<00:00, 171.00it/s]
Encrypting: 100%|█████████████████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 155.25it/s] 
Client 1 Epoch 1/20 | Loss: 0.1023 | Acc: 96.24%
Client 1 Epoch 2/20 | Loss: 0.0884 | Acc: 96.62%
Client 1 Epoch 3/20 | Loss: 0.0794 | Acc: 97.58%
Client 1 Epoch 4/20 | Loss: 0.0691 | Acc: 97.71%
Client 1 Epoch 5/20 | Loss: 0.0676 | Acc: 97.39%
Client 1 Epoch 6/20 | Loss: 0.0598 | Acc: 97.64%
Client 1 Epoch 7/20 | Loss: 0.0650 | Acc: 97.71%
Client 1 Epoch 8/20 | Loss: 0.0522 | Acc: 98.09%
Client 1 Epoch 9/20 | Loss: 0.0736 | Acc: 98.02%
Client 1 Epoch 10/20 | Loss: 0.0570 | Acc: 97.90%
Client 1 Epoch 11/20 | Loss: 0.0591 | Acc: 97.64%
Client 1 Epoch 12/20 | Loss: 0.0433 | Acc: 98.47%
Client 1 Epoch 13/20 | Loss: 0.0388 | Acc: 98.66%
Client 1 Epoch 14/20 | Loss: 0.0494 | Acc: 98.41%
Client 1 Epoch 15/20 | Loss: 0.0362 | Acc: 98.60%
Client 1 Epoch 16/20 | Loss: 0.0365 | Acc: 98.47%
Client 1 Epoch 17/20 | Loss: 0.0314 | Acc: 99.11%
Client 1 Epoch 18/20 | Loss: 0.0345 | Acc: 98.66%
Client 1 Epoch 19/20 | Loss: 0.0299 | Acc: 98.79%
Client 1 Epoch 20/20 | Loss: 0.0270 | Acc: 99.24%
Encrypting: 100%|███████████████████████████████████████████████████████████████████████| 100352/100352 [07:40<00:00, 217.91it/s] 
Encrypting: 100%|█████████████████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 233.64it/s] 
Client 2 Epoch 1/20 | Loss: 0.0926 | Acc: 96.49%
Client 2 Epoch 2/20 | Loss: 0.0978 | Acc: 96.37%
Client 2 Epoch 3/20 | Loss: 0.0780 | Acc: 97.07%
Client 2 Epoch 4/20 | Loss: 0.0802 | Acc: 96.81%
Client 2 Epoch 5/20 | Loss: 0.0620 | Acc: 97.90%
Client 2 Epoch 6/20 | Loss: 0.0710 | Acc: 97.58%
Client 2 Epoch 7/20 | Loss: 0.0702 | Acc: 97.64%
Client 2 Epoch 8/20 | Loss: 0.0621 | Acc: 97.96%
Client 2 Epoch 9/20 | Loss: 0.0519 | Acc: 97.90%
Client 2 Epoch 10/20 | Loss: 0.0478 | Acc: 98.15%
Client 2 Epoch 11/20 | Loss: 0.0338 | Acc: 98.98%
Client 2 Epoch 12/20 | Loss: 0.0417 | Acc: 98.73%
Client 2 Epoch 13/20 | Loss: 0.0359 | Acc: 98.53%
Client 2 Epoch 14/20 | Loss: 0.0327 | Acc: 99.11%
Client 2 Epoch 15/20 | Loss: 0.0373 | Acc: 98.98%
Client 2 Epoch 16/20 | Loss: 0.0342 | Acc: 98.60%
Client 2 Epoch 17/20 | Loss: 0.0245 | Acc: 99.43%
Client 2 Epoch 18/20 | Loss: 0.0212 | Acc: 99.11%
Client 2 Epoch 19/20 | Loss: 0.0354 | Acc: 98.79%
Client 2 Epoch 20/20 | Loss: 0.0160 | Acc: 99.62%
Encrypting: 100%|███████████████████████████████████████████████████████████████████████| 100352/100352 [07:11<00:00, 232.38it/s]
Encrypting: 100%|█████████████████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 233.18it/s] 
Decrypting: 100%|███████████████████████████████████████████████████████████████████████| 100352/100352 [01:57<00:00, 852.56it/s]
Decrypting: 100%|█████████████████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 777.83it/s]
Client 0 Val Acc: 95.23%
Client 1 Val Acc: 94.27%
Client 2 Val Acc: 95.99%
Round Average Val Acc: 95.17%

=== Round 3/3 ===
Client 0 Epoch 1/20 | Loss: 0.0820 | Acc: 97.07%
Client 0 Epoch 2/20 | Loss: 0.0662 | Acc: 97.71%
Client 0 Epoch 3/20 | Loss: 0.0633 | Acc: 98.09%
Client 0 Epoch 4/20 | Loss: 0.0610 | Acc: 97.83%
Client 0 Epoch 5/20 | Loss: 0.0761 | Acc: 97.45%
Client 0 Epoch 6/20 | Loss: 0.0479 | Acc: 98.28%
Client 0 Epoch 7/20 | Loss: 0.0602 | Acc: 97.96%
Client 0 Epoch 8/20 | Loss: 0.0449 | Acc: 98.41%
Client 0 Epoch 9/20 | Loss: 0.0545 | Acc: 97.64%
Client 0 Epoch 10/20 | Loss: 0.0310 | Acc: 98.98%
Client 0 Epoch 11/20 | Loss: 0.0361 | Acc: 98.92%
Client 0 Epoch 12/20 | Loss: 0.0327 | Acc: 98.73%
Client 0 Epoch 13/20 | Loss: 0.0400 | Acc: 98.66%
Client 0 Epoch 14/20 | Loss: 0.0199 | Acc: 99.49%
Client 0 Epoch 15/20 | Loss: 0.0304 | Acc: 99.24%
Client 0 Epoch 16/20 | Loss: 0.0217 | Acc: 99.30%
Client 0 Epoch 17/20 | Loss: 0.0257 | Acc: 99.17%
Client 0 Epoch 18/20 | Loss: 0.0200 | Acc: 99.49%
Client 0 Epoch 19/20 | Loss: 0.0271 | Acc: 99.17%
Client 0 Epoch 20/20 | Loss: 0.0276 | Acc: 99.04%
Encrypting: 100%|███████████████████████████████████████████████████████████████████████| 100352/100352 [09:00<00:00, 185.79it/s]
Encrypting: 100%|█████████████████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 146.83it/s] 
Client 1 Epoch 1/20 | Loss: 0.0705 | Acc: 97.32%
Client 1 Epoch 2/20 | Loss: 0.0581 | Acc: 97.58%
Client 1 Epoch 3/20 | Loss: 0.1349 | Acc: 98.02%
Client 1 Epoch 4/20 | Loss: 0.0476 | Acc: 98.09%
Client 1 Epoch 5/20 | Loss: 0.0524 | Acc: 98.28%
Client 1 Epoch 6/20 | Loss: 0.0485 | Acc: 98.22%
Client 1 Epoch 7/20 | Loss: 0.0355 | Acc: 98.92%
Client 1 Epoch 8/20 | Loss: 0.0379 | Acc: 98.85%
Client 1 Epoch 9/20 | Loss: 0.0395 | Acc: 98.66%
Client 1 Epoch 10/20 | Loss: 0.0245 | Acc: 99.17%
Client 1 Epoch 11/20 | Loss: 0.0330 | Acc: 98.53%
Client 1 Epoch 12/20 | Loss: 0.0296 | Acc: 98.92%
Client 1 Epoch 13/20 | Loss: 0.0339 | Acc: 98.92%
Client 1 Epoch 14/20 | Loss: 0.0229 | Acc: 99.24%
Client 1 Epoch 15/20 | Loss: 0.0223 | Acc: 99.17%
Client 1 Epoch 16/20 | Loss: 0.0228 | Acc: 99.04%
Client 1 Epoch 17/20 | Loss: 0.0283 | Acc: 99.17%
Client 1 Epoch 18/20 | Loss: 0.0176 | Acc: 99.62%
Client 1 Epoch 19/20 | Loss: 0.0147 | Acc: 99.49%
Client 1 Epoch 20/20 | Loss: 0.0158 | Acc: 99.55%
Encrypting: 100%|███████████████████████████████████████████████████████████████████████| 100352/100352 [11:04<00:00, 150.99it/s]
Encrypting: 100%|█████████████████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 159.25it/s] 
Client 2 Epoch 1/20 | Loss: 0.0535 | Acc: 98.34%
Client 2 Epoch 2/20 | Loss: 0.0650 | Acc: 97.96%
Client 2 Epoch 3/20 | Loss: 0.0625 | Acc: 97.71%
Client 2 Epoch 4/20 | Loss: 0.0424 | Acc: 98.41%
Client 2 Epoch 5/20 | Loss: 0.0541 | Acc: 97.71%
Client 2 Epoch 6/20 | Loss: 0.0344 | Acc: 98.60%
Client 2 Epoch 7/20 | Loss: 0.0493 | Acc: 98.53%
Client 2 Epoch 8/20 | Loss: 0.0310 | Acc: 99.04%
Client 2 Epoch 9/20 | Loss: 0.0369 | Acc: 98.60%
Client 2 Epoch 10/20 | Loss: 0.0322 | Acc: 98.73%
Client 2 Epoch 11/20 | Loss: 0.0249 | Acc: 98.98%
Client 2 Epoch 12/20 | Loss: 0.0313 | Acc: 98.79%
Client 2 Epoch 13/20 | Loss: 0.0298 | Acc: 98.79%
Client 2 Epoch 14/20 | Loss: 0.0151 | Acc: 99.49%
Client 2 Epoch 15/20 | Loss: 0.0128 | Acc: 99.49%
Client 2 Epoch 16/20 | Loss: 0.0347 | Acc: 98.98%
Client 2 Epoch 17/20 | Loss: 0.0141 | Acc: 99.62%
Client 2 Epoch 18/20 | Loss: 0.0089 | Acc: 99.75%
Client 2 Epoch 19/20 | Loss: 0.0183 | Acc: 99.62%
Client 2 Epoch 20/20 | Loss: 0.0164 | Acc: 99.62%
Encrypting: 100%|███████████████████████████████████████████████████████████████████████| 100352/100352 [08:23<00:00, 199.17it/s]
Encrypting: 100%|█████████████████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 230.77it/s] 
Decrypting: 100%|███████████████████████████████████████████████████████████████████████| 100352/100352 [01:58<00:00, 848.05it/s]
Decrypting: 100%|█████████████████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 666.46it/s]
Client 0 Val Acc: 95.42%
Client 1 Val Acc: 95.80%
Client 2 Val Acc: 95.42%
Round Average Val Acc: 95.55%
个性层差异
=======================
个性化层平均差异度：10.9964

(pytorch_gpu) C:\Users\coolboy\Desktop\deep_study\Medmodel_cry>
