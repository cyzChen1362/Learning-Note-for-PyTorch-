"""
torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
torchvision.utils: 其他的一些有用的方法。
"""

import torch
import torchvision
import numpy as np
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l
from matplotlib import pyplot as plt

"""
    softmax回归的从零开始实现
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
# 2. 初始化模型参数
# =======================
# 每张图片都是28*28=784个像素，每个像素都是一个值，所以输入端有784个输入
num_inputs = 784
# 每张图片最后分为10个类别中的一个，所以有10个类别概率输出
num_outputs = 10
# num_inputs个输入，num_outputs个输出，参数矩阵自然是 num_inputs * num_outputs
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
# 偏置项同理
b = torch.zeros(num_outputs, dtype=torch.float)
# 打开模型参数梯度运算
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# =======================
# 3. 实现softmax运算
# =======================
# 输入X是一个矩阵，每行是一个样本
def softmax(X):
    # 各元素求指数
    X_exp = X.exp()
    # 对列求和，得出每行的指数和，形成n行1列
    partition = X_exp.sum(dim=1, keepdim=True)
    # partition应用广播机制，按列复制，变成n行10列，对应的元素再除
    return X_exp / partition

# test softmax(X)
X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(dim=1, keepdim=True))

# =======================
# 4. 定义模型
# =======================
def net(X):
    # torch.mm：做矩阵乘法，最终返回n行10列softmax结果
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

# =======================
# 5. 定义损失函数
# =======================
"""
# 预测概率张量
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# 真实的标签分别是第0类和第2类
y = torch.LongTensor([0, 2])

# y变成2行1列
# gather(dim=1, index)：在每一行中按dim=1列检索，根据index的位置取出对应值
y_hat.gather(1, y.view(-1, 1))

输出：
tensor([[0.1000],
        [0.5000]])
        
"""

# 交叉熵函数
def cross_entropy(y_hat, y):
    # 每一行都是 -log(yj_hat)
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

# =======================
# 6. 计算分类准确率
# =======================
"""
def accuracy(y_hat, y):
    # y_hat.argmax(dim=1)返回矩阵y_hat每行中最大元素的索引，且返回结果与变量y形状相同
    # float()将判断返回布尔值变成0/1，然后mean即可得出正确率
    return (y_hat.argmax(dim=1) == y).float().mean().item()
print(accuracy(y_hat, y))

输出：0.5

"""

# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进；
# 它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    # 初始化准确率和batch数
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        # net(X)返回n行10列的softmax结果，y为真实值；
        # 每一个for循环的data_iter是怎么处理的，详见第一部分注释
        # argmax(dim=1)沿列找出每行最大值，变成n行1列
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        # 叠加batch数（即真实值标签数）
        n += y.shape[0]
    return acc_sum / n

# =======================
# 7. 训练模型
# =======================

# 更新num_epochs次梯度，学习率lr
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh包中方便以后使用
# net自定义模型；data_iter数据集迭代器；loss损失函数；num_epochs训练总轮数；batch_size每批样本数
# optimizer手写sgd或自动
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        # 初始化训练损失函数/准确率各批次之和 && 总样本数为0
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            # 模型预测一次y_hat
            y_hat = net(X)
            # 这一批次的所有损失函数之和
            # 这里的loss是某批次所有损失函数的和
            # nn.CrossEntropyLoss()得到的loss是批次的平均
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
                optimizer.step()  # “softmax回归的简洁实现”一节将用到

            # 这里是训练集的损失函数/准确率
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        # 这里是测试集的准确率
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# =======================
# 8. 预测
# =======================
# iter(test_iter)将test_iter转换成一个迭代器对象；
# next()取出第一个batch
X, y = iter(test_iter).next()
# 将真实标签y和模型预测结果net(X)的标签从数字转为对应的文字标签
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
# 标题
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
# 作图
d2l.show_fashion_mnist(X[0:9], titles[0:9])
plt.show()



