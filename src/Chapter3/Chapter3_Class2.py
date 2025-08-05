import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

"""
    线性回归的从零开始实现
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
# 2. 生成第二个特征和标签的散点图
# =======================
# def use_svg_display():
#     # 用矢量图显示
#     display.set_matplotlib_formats('svg')
#
# def set_figsize(figsize = (3.5,2.5)):
#     use_svg_display()
#     # 设置图的尺寸
#     plt.rcParams['figure.figsize'] = figsize
#
# set_figsize()
# plt.scatter(features[:,1].numpy(), labels.numpy(),1)
# plt.show()

# =======================
# 3. 读取数据
# =======================
# 小批量读取器：每次随机打乱数据顺序，并按batch_size返回样本
def data_iter(batch_size, features, labels):
    # len(tensor)返回第0维的长度，也就是行数
    num_examples = len(features)
    # 生成索引列表
    indices = list(range(num_examples))
    # 样本的读取顺序是随机的
    random.shuffle(indices)
    # 注意这里步长是batch_size，不是步长为1
    for i in range(0, num_examples, batch_size):
        # 索引行数j = indices索引列表中的第i行到第i+batch_size行，并进行类型转换
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        # 在第0维索引行数为j（j是一个列表而不是一个数）的这些样本，提取出来，yield返回这两个提取值，并且省内存
        yield  features.index_select(0, j), labels.index_select(0, j)

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    # 输出一个batch_size就停了
    break

# =======================
# 4.初始化模型参数
# =======================
# 将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0
# (num_inputs, 1)即生成形状为num_inputs行1列
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
# 设定可求梯度
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# =======================
# 5.定义模型
# =======================
def linreg(X, w, b):
    # torch.mm做矩阵乘法
    return torch.mm(X, w) + b

# =======================
# 6.定义损失函数
# =======================
def squared_loss(y_hat, y):
    # 把真实值y变形成预测值y_hat的形状
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

# =======================
# 7.定义优化算法
# =======================
def sgd(params, lr, batch_size):
    # params为θ；param为w,b；lr为learning rate；
    for param in params:
        # 由于是一个批次求一次loss并相加，然后在算梯度的，所以会除以batch_size
        # 注意这里更改param时用的param.data，直接修改参数，并且这个操作不会记录在反向传播里面
        param.data -= lr * param.grad / batch_size

# =======================
# 8.训练模型
# =======================
lr = 0.01
num_epochs = 10
net = linreg
loss = squared_loss

# 训练模型一共需要num_epochs个迭代周期
for epoch in range(num_epochs):
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）
    # X和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        # net是算出来的y_hat，l是有关小批量X和y的损失，有个sum
        # 这里的sum就是sgd里面除以batch_size的原因
        l = loss(net(X, w, b), y).sum()
        # 小批量的损失对模型参数求梯度
        l.backward()
        # 使用小批量随机梯度下降迭代模型参数
        sgd([w, b], lr, batch_size)
        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    # train_l是一列的损失函数向量，用第epoch+1步得出的w,b来算
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

# 训练完成后，比较参数
print(true_w, '\n', w)
print(true_b, '\n', b)










