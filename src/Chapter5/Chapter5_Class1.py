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
# 卷积运算其实就是互相关运算的核函数上下左右翻转了而已
# 深度学习中用的其实都是互相关运算，只是习惯仍然叫做卷积而已
# 实际上也一样，因为参数都是学回来的，如果核函数翻转的同时输入也翻转，输出将不变
#
# 特征图就是输出结果
# 感受野就是卷积核能覆盖到的区域
# 加多几层，感受野就大一点嘛
# ********************************************************************************

"""
    二维卷积层
"""

import torch
from torch import nn

def corr2d(X, K):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    # 核函数的形状
    h, w = K.shape
    # 输出的形状
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    # 互相关运算
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K))

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        # 父函数初始化
        super(Conv2D, self).__init__()
        # 参数初始化
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# ======================================
# 图像中物体边缘检测 + 通过数据学习核数组
# ======================================

X = torch.ones(6, 8)
X[:, 2:6] = 0

K = torch.tensor([[1, -1]])

Y = corr2d(X, K)
print(Y)

# 构造一个核数组形状是(1, 2)的二维卷积层
conv2d = Conv2D(kernel_size=(1, 2))

step = 100
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清0
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))

print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.bias.data)




