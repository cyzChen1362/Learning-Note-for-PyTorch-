import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

"""
    softmax回归的简洁实现
"""

# =======================
# 1. 获取和读取数据
# =======================
batch_size = 256
# 加载训练集和测试集
# 这里的data_iter是一个数据迭代器，按照batch_size每次迭代自动返回一个batch
# 每个 batch 是一个元组 (X, y)，X 是数据，y 是标签
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# =======================
# 2. 定义和初始化模型
# =======================
# 每张图片都是28*28=784个像素，每个像素都是一个值，所以输入端有784个输入
num_inputs = 784
# 每张图片最后分为10个类别中的一个，所以有10个类别概率输出
num_outputs = 10

"""

# LinearNet的结构复习：

# 定义了一个名为 LinearNet 的神经网络类，继承自 nn.Module
class LinearNet(nn.Module):
    # 初始化方法
    def __init__(self, num_inputs, num_outputs):
        # 调用父类（nn.Module）的构造函数，完成基类初始化（必须写）
        super(LinearNet, self).__init__()
        # 定义一个线性层，输入维度为 num_inputs，输出维度为 num_outputs
        # nn.Linear 是 PyTorch 的线性变换模块，自动包含权重和偏置
        self.linear = nn.Linear(num_inputs, num_outputs)
    # 注意：这里的forward方法是写死了的；
    # 一旦调用net(X),进入net的LinearNet层，就会自动调用forward
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        # view()相当于把行数变为batch，列数变为1*28*28
        y = self.linear(x.view(x.shape[0], -1))
        return y

net = LinearNet(num_inputs, num_outputs)

"""

# 上文注释中，对x的形状转换的这个功能自定义一个FlattenLayer层
# 仍然是继承自父类nn.Module
# 本函数已保存在d2lzh_pytorch包中方便以后使用
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    # 注意：这里的forward方法是写死了的；
    # 一旦调用net(X),进入net的FlattenLayer层，就会自动调用forward
    # 这里一旦名字不是forward就报错
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

from collections import OrderedDict

net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        # 第一层FlattenLayer层
        # 将X[batch,1,28,28]展平为[batch,1*28*28]
        ('flatten', FlattenLayer()),
        # 第二层Linear层
        # 承接第一层的输出作为第二层的输入
        # 将输入[batch,1*28*28]变为输出[batch,10]（这里提前设置好了）
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

"""
如果想要第二层（linear层）的输入维度自动适配第一层（flatten层）的输出维度，
也就是不手动写死 num_inputs = 784，
可以使用nn.LazyLinear（适用于 PyTorch 1.8 及以上版本）

from collections import OrderedDict
from torch import nn

net = nn.Sequential(OrderedDict([
    ('flatten', FlattenLayer()),
    ('linear', nn.LazyLinear(out_features=10))  # 自动推断 in_features
]))

"""

# 初始化参数
# normal_表示把权重初始化为正态分布的随机数
# constant_表示把偏置初始化为固定的常数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# =======================
# 3. softmax和交叉熵损失函数
# =======================
"""
import torch
import torch.nn.functional as F

logits = torch.tensor([[10000.0, 1.0, -10000.0]])
labels = torch.tensor([0])

# 自己分开写可能爆炸
probs = torch.softmax(logits, dim=1)
loss = -torch.log(probs[0][labels])
print("分开算的 loss:", loss)  # 可能是 inf 或 nan

# PyTorch推荐的做法
loss_stable = F.cross_entropy(logits, labels)
print("内置 F.cross_entropy:", loss_stable)  # 正常值

"""
# 如果分开定义softmax运算和交叉熵损失函数可能会造成数值不稳定，例如exp(-1000)可能会过小
# 因此使用这个函数
loss = nn.CrossEntropyLoss()

# =======================
# 4. 定义优化算法
# =======================
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# =======================
# 5. 训练模型
# =======================
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

"""
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        # 初始化训练损失函数/准确率各批次之和 && 总样本数为0
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            # 模型预测一次y_hat
            y_hat = net(X)
            # 这一批次的所有损失函数之和
            
            # ************************************************************
            # 这里比较有意思的一点是：
            # 在3.6的loss的输入是已经softmax的结果，以及标签真实值对应索引
            # 根据索引取出softmax对应位置y_hat进行log；
            # 但是这里的loss输入是原始的未softmax数据
            # 然后使用nn.CrossEntropyLoss计算
            # ************************************************************
            
            l = loss(y_hat, y).sum()

            # 梯度清零
            # .zero_grad()不是optimizer的方法，而是param的方法；
            # param=torch.tensor()时自带.grad属性和.grad.zero_()的能力
            # 如果自定义了优化器，那么梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            # 如果没自定义优化器且有参数且参数有梯度，那么参数梯度清零
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            # 打开反向传播，自动求导计算梯度（注意l是一个标量）
            l.backward()
            if optimizer is None:
                # param.data -= lr * param.grad / batch_size
                d2l.sgd(params, lr, batch_size)
            else:
            
                # ************************************************************
                # 这里比较有意思的另一点是：
                # optimizer的step()方法继承自：
                # PyTorch 的 优化器类（例如 torch.optim.SGD、Adam 等）的方法
                # 如果是自己写的optimizer，并不会自带step()方法
                # 当然，可以考虑这样写自己的optimizer：
                # class MySGD:
                #     def __init__(self, params, lr):
                #         self.params = params
                #         self.lr = lr
                # 
                #     def step(self, batch_size):
                #         for param in self.params:
                #             param.data -= self.lr * param.grad / batch_size
                # 
                #     def zero_grad(self):
                #         for param in self.params:
                #             if param.grad is not None:
                #                 param.grad.zero_()
                # ************************************************************
            
                optimizer.step()

            # 这里是训练集的损失函数/准确率
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        # 这里是测试集的准确率
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

"""






