"""
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


import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

"""
    线性回归的简洁实现
"""

# =======================
# 1. 生成训练数据集
# =======================
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
# 生成example行，inputs列的矩阵，元素为标准正态分布，作为特征的值
features = torch.randn(num_examples, num_inputs,dtype=torch.float32)
# 生成无噪声标签，会广播的
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
# 添加噪声，均值0，标准差0.01，大小和标签一致
labels += torch.tensor(np.random.normal(0,0.01,size = labels.size()),dtype=torch.float32)

# =======================
# 2. 生成训练数据集
# =======================
import torch.utils.data as Data

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break

"""
注：此处data_iter同：

def data_iter(batch_size, features, labels):
    # len(tensor)返回第0维的长度，也就是行数
    num_examples = len(features)
    # 生成索引列表
    indices = list(range(num_examples))
    # 样本的读取顺序是随机的
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # 索引行数j = indices索引列表中的第i行到第i+batch_size行，并进行类型转换
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        # 在第0维索引行数为j的这些样本，提取出来，yield返回这两个提取值，并且省内存
        yield  features.index_select(0, j), labels.index_select(0, j)
"""

# =======================
# 3. 定义模型
# =======================
import torch.nn as nn

# 定义了一个名为 LinearNet 的神经网络类，继承自 nn.Module
class LinearNet(nn.Module):
    # 初始化方法，n_feature 表示输入特征的数量
    def __init__(self, n_feature):
        # 调用父类（nn.Module）的构造函数，完成基类初始化（必须写）
        super(LinearNet, self).__init__()
        # 定义一个线性层，输入维度为 n_feature，输出为 1
        # nn.Linear 是 PyTorch 的线性变换模块，自动包含权重和偏置
        self.linear = nn.Linear(n_feature, 1)
    # forward 定义前向传播
    # 必须用这个名字
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net) # 使用print可以打印出网络的结构

"""
或者也可以通过更简单的方式搭建网络：
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])

输出：
Sequential(
  (linear): Linear(in_features=2, out_features=1, bias=True)
)
Linear(in_features=2, out_features=1, bias=True)
"""

# =======================
# 4. 初始化模型参数
# =======================
from torch.nn import init

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net.bias.data.fill_(0)

# =======================
# 5. 定义损失函数
# =======================
# nn.MSELoss()是标准的均方误差，没有1/2
loss = nn.MSELoss()

# =======================
# 6. 定义优化算法
# =======================
import torch.optim as optim

# 虽然自己没写parameters()方法，但继承的nn.Module有这个方法
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

"""
注：此处sgd类似（但没除以batch_size）：

def sgd(params, lr, batch_size):
    # params为θ；param为w,b；lr为learning rate；
    for param in params:
        # 由于是一个批次求一次loss并相加，然后在算梯度的，所以会除以batch_size
        # 注意这里更改param时用的param.data，直接修改参数，并且这个操作不会记录在反向传播里面
        param.data -= lr * param.grad / batch_size


为不同子网络设置不同学习率的方法：

# optim.SGD 可以传入一个列表，列表中的每一项是一个字典（dict），每个字典对应一个“参数组”
# 每个“参数组”都可以单独设置学习率（lr），也可以设置其他优化器参数（如 momentum 等）
optimizer =optim.SGD([
                # 如果对某个参数不指定学习率，就使用最外层的默认学习率
                {
                    # 参数：net.subnet1.parameters()，表示subnet1的所有可训练参数
                    # 学习率：未指定lr，所以使用外层的lr=0.03
                    'params': net.subnet1.parameters()}, 
                {
                    # 参数：net.subnet2.parameters()，表示subnet2的所有可训练参数
                    # 学习率：指定为0.01，覆盖了外层默认值
                    'params': net.subnet2.parameters(), 'lr': 0.01}
            ], lr=0.03)
            
什么叫一个网络有两个子网络呢？例如：

class MyNet(nn.Module):
    def __init__(self):
        # 一个网络有两个子网络
        super().__init__()
        # 一个子网络有两层，第一层是一个线性回归层（全链接层），把10维映射成5维
        # 第二层是一个ReLU，非线性激活层
        self.subnet1 = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        self.subnet2 = nn.Sequential(nn.Linear(5, 2), nn.ReLU())

net = MyNet()

调整学习率：

for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1 # 学习率为之前的0.1倍

"""

# =======================
# 7. 训练模型
# =======================
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        # view中-1表示这一维度自动推断
        l = loss(output, y.view(-1, 1))
        # 梯度清零，等价于net.zero_grad()
        optimizer.zero_grad()
        l.backward()
        # optimizer由SGD方法创建，使用optimizer.step()方法即学习参数
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch,l.item()))

# 访问参数
dense = net
print(true_w, dense.linear.weight)
print(true_b, dense.linear.bias)
