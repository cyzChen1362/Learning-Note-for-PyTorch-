r"""
学习笔记

原始工程来源：
    ShusenTang / Dive-into-DL-PyTorch
    仓库地址：https://github.com/ShusenTang/Dive-into-DL-PyTorch

原始文献引用：
    @book{zhang2019dive,
        title={Dive into Deep Learning},
        author={Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola},
        note={\url{http://www.d2l.ai}},
        year={2020}
    }

用途说明：
    本文件基于该工程，加入了个人理解与注释，用作学习笔记，不用于商业用途。

许可协议：
    原工程遵循 Apache-2.0 许可证。:contentReference[oaicite:1]{index=1}
"""

# ********************************************************************************
# 在小批量随机梯度下降中，提到的方差指的是：
# 在全样本中抽取的小批量，其样本的分布总会和全样本有些差别
# 这个差别也就是所谓的抽样噪声
# 所以说，使用小批量随机梯度下降，总会引入一些噪声
# 但这也可以防止过拟合，添加了一些正则化的功能
#
# 另一个点是课本中提到的“每个小批量梯度里可能含有更多的冗余信息”
# 例如一个全样本，里面有10个样本完全一致
# 那你算这10个完全一样的样本的梯度，就浪费一些资源了
# 而且这10个样本用同样的方式告诉你梯度怎样变化，相当于只有一个样本的功效
#
# 例如3个样本，两个样本告诉你向左，一个样本告诉你向下
# 如果是全批量，就相当于告诉你向左下移动根号五的距离
# 但如果是随机梯度下降
# 就相当于告诉你先往左移动一，再往左移动一，最后往下移动一，一共移动了三的距离
# 所以随机梯度下降在样本有相同的情况下，移动的距离会更长
# 也就是所谓的探索更多，更有可能绕开障碍
# ********************************************************************************

"""
    小批量随机梯度下降
"""

# ========================
# 读取数据
# ========================

import numpy as np
import time
import torch
from torch import nn, optim
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

def get_data_ch7():  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    data = np.genfromtxt('../../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
    torch.tensor(data[:1500, -1], dtype=torch.float32) # 前1500个样本(每个样本5个特征)

features, labels = get_data_ch7()
print(features.shape) # torch.Size([1500, 5])
print(labels.shape)   # torch.Size([1500])

# ========================
# 从零开始实现
# ========================

# 超参数字典hyperparams
def sgd(params, states, hyperparams):
    for p in params:
        # .data防止优化的过程也进入了梯度计算的计算图中
        p.data -= hyperparams['lr'] * p.grad.data

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch7(optimizer_fn, states, hyperparams, features, labels,
              batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = d2l.linreg, d2l.squared_loss

    # 最终的输出为一个标签，那 w 自然是 input_num 行 1 列
    w = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32),
                           requires_grad=True)
    # 输出就一个，那偏置也就一个呗
    b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    # 衡量损失函数的损失值
    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()

    # 这里先用一次，计算出还没开始迭代的网络的损失是多少，作为起始的第一个损失
    ls = [eval_loss()]
    # 将特征和标签整合成一个可迭代对象
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        # enumerate 会在迭代时自动提供序号，从 0 开始计数
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X, w, b), y).mean()  # 使用平均损失

            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            # 反向传播
            l.backward()
            optimizer_fn([w, b], states, hyperparams)  # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # 每100个样本记录下当前训练误差
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    d2l.plt.show()

def train_sgd(lr, batch_size, num_epochs=2):
    train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)

# 梯度下降
train_sgd(1, 1500, 6)

# 随机梯度下降
train_sgd(0.005, 1)

# 小批量随机梯度下降
train_sgd(0.05, 10)

# ========================
# 简洁实现
# ========================

# 通过创建optimizer实例来调用优化算法

# 本函数与原书不同的是这里第一个参数优化器函数而不是优化器的名字
# 例如: optimizer_fn=torch.optim.SGD, optimizer_hyperparams={"lr": 0.05}
def train_pytorch_ch7(optimizer_fn, optimizer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    # 初始化模型，省下的主要是自己写d2l的步骤
    net = nn.Sequential(
        nn.Linear(features.shape[-1], 1)
    )
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            # 除以2是为了和train_ch7保持一致, 因为squared_loss中除了2
            l = loss(net(X).view(-1), y) / 2

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    d2l.plt.show()

train_pytorch_ch7(optim.SGD, {"lr": 0.05}, features, labels, 10)



