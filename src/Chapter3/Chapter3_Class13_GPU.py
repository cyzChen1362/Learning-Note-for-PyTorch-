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
    丢弃法实验_GPU
"""

import time
start = time.time()

# =======================
# 0. 选择设备
# =======================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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
    mask = (torch.rand(X.shape, device=X.device) < keep_prob).float()
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
W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)),
                  dtype=torch.float, requires_grad=True, device=device)
b1 = torch.zeros(num_hiddens1, requires_grad=True, device=device)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)),
                  dtype=torch.float, requires_grad=True, device=device)
b2 = torch.zeros(num_hiddens2, requires_grad=True, device=device)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)),
                  dtype=torch.float, requires_grad=True, device=device)
b3 = torch.zeros(num_outputs, requires_grad=True, device=device)
# 参数列表
params = [W1, b1, W2, b2, W3, b3]

# =======================
# 3. 定义模型
# =======================

drop_prob1, drop_prob2 = 0.2, 0.5

# 网络
def net(X, is_training=True):
    X = X.view(-1, num_inputs).to(device)
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

# 准确率评估函数
# 本函数已保存在d2lzh_pytorch
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        # 判断这个net是不是torch.nn.Module类型或子类
        if isinstance(net, torch.nn.Module):
            # 如果是，那么就会有eval()和train()方法
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train() # 改回训练模式
        else: # 自定义的模型
            if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                # 将is_training设置成False
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
            else:
                # 否则就不管了，直接算（...）
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

# =======================
# 4. 训练和测试模型
# =======================

num_epochs, lr, batch_size = 5, 100.0, 256
# 图片分类，用softmax+交叉熵，即batch平均损失CrossEntropyLoss
# 同样，CrossEntropyLoss和train_ch3中的sgd都有除以batch，所以lr得乘大一点
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 这里d2l.train_ch3改一下，加上.to(device)

def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到
            
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)


# =======================
# 5. 简洁实现
# =======================
# 这里torch也直接内置了Dropout层
# nn.Sequential 本质上就是 torch.nn.Module 的一个子类
# 所以会有net.eval()和net.train()方法
net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(drop_prob1),
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        nn.Dropout(drop_prob2),
        nn.Linear(num_hiddens2, 10)
        )

net = net.to(device)
for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

# 训练并测试模型
# torch的SGD不会除以batch，和CrossEntropyLoss刚好搭配
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)


end = time.time()
print(f"训练用时：{end - start:.2f} 秒")
