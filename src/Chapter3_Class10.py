import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

"""
    多层感知机的简洁实现
"""

# =======================
# 1. 定义模型
# =======================
# 输入个数1*28*28，输出个数10，隐藏层单元数256
num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
        # 第一层FlattenLayer展平层
        d2l.FlattenLayer(),
        # 第二层隐藏层，Linear + ReLU
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        # 第三次输出层，Linear
        nn.Linear(num_hiddens, num_outputs),
        )

# 初始化参数，全部都正态分布
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

# =======================
# 2. 读取数据并训练模型
# =======================
# 获取和读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 损失函数
loss = torch.nn.CrossEntropyLoss()

# 优化器，这里使用的是PyTorch的SGD而不是d2lzh_pytorch里面的sgd（见下面）
#
# def sgd(params, lr, batch_size):
#     # params为θ；param为w,b；lr为learning rate；
#     for param in params:
#         # 就是指这里的batch_size
#         param.data -= lr * param.grad / batch_size
#
# 所以这里学习率就没有问题很大，不用再乘以batch_size

# pytorch-SGD优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

# 轮次为5
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

