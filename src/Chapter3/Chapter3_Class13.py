# ********************************************************************************
# 以下是对过拟合/正则化/丢弃法的理解：
# 例如有784个特征的图片输入，中间隐藏层有256个神经元
# 那么就是希望256个神经元各自学习图片的不同特征
#
# 所谓过拟合，就是只有少数神经元的参数特别发达。其余神经元的参数都接近于0
# 训练集确实可以根据少数参数发达的神经元算出很小的loss以及很好的准确率
# 但这只是学习到了训练集的少数几个共有特征，迁移到测试集就很差了
#
# 那么正则化就是将参数往0的方向调整，参数越大惩罚越大
# 这就防止了某少数几个神经元的参数越滚越大
#
# 丢弃法的思想就是：
# 在每个batch训练/迭代参数的时候，都随机屏蔽不同的神经元
# 更严谨来说，就是在前向传播的时候，随即几个神经元的输出清零，但参数是不清零的
# 反向传播的时候，同样这几个被屏蔽的神经元也是不会更新的
# 例如屏蔽了几个参数发达的神经元，剩下一堆0参数的神经元就会有大的参数更新
# 多次这样迭代，就会多个神经元都用上，极大减轻了参数分布不均的情况了
# ********************************************************************************

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

"""
    丢弃法实验
"""

# =======================
# 1. dropout函数
# =======================

def dropout(X, drop_prob):
    # 张量转为浮点数
    X = X.float()
    # 确定drop_prob在0~1之间，否则报错
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    # 这里mask的思路是：
    # 生成一个X同形张量，元素的值为0~1，然后与keep_prob判断
    # 小于keep_prob的为Ture-->1，大于为0
    # 这样mask就是一个由0/1组成的张量，就得出了是否保留参数的分布
    mask = (torch.rand(X.shape) < keep_prob).float()
    # 通过mask乘以张量X，实现0的地方不保留，最后拉伸一下
    return mask * X / keep_prob

# test
# X = torch.arange(16).view(4, 4)
# print(dropout(X, 0))
# print(dropout(X, 0.5))
# print(dropout(X, 1))

# =======================
# 2. 定义模型参数
# =======================
# 使用Fashion-MNIST数据集
# 定义一个包含两个隐藏层的多层感知机
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
# 随机初始化参数
W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)
# 参数列表
params = [W1, b1, W2, b2, W3, b3]

# =======================
# 3. 定义模型
# =======================

drop_prob1, drop_prob2 = 0.2, 0.5

def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training:  # 只在训练模型时使用丢弃法
        # 丢弃第一个隐藏层的部分输出
        H1 = dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        # 丢弃第二个隐藏层的部分输出
        H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层
    # 返回值为输出层输出
    return torch.matmul(H2, W3) + b3













