r"""
学习笔记

原始工程来源：
    D2L (Dive into Deep Learning) 中文版
    仓库地址：https://github.com/d2l-ai/d2l-zh
    官方网站：https://zh.d2l.ai/

原始文献引用：
    @book{zhang2019dive,
        title={Dive into Deep Learning},
        author={Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola},
        note={\url{https://zh.d2l.ai/}},
        year={2020}
    }

用途说明：
    本文件基于《动手学深度学习》中文版（d2l-zh）及其代码进行学习与注释，
    仅作个人学习笔记与交流之用，不用于商业用途。

许可协议：
    原工程遵循 Apache-2.0 许可证。
"""

"""
    转置卷积
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_computer-vision/transposed-conv.html
# ********************************************************************************

# ********************************************************************************
# 这一节大概就是一些基本概念
# 这里仅对一些代码做部分注释
# 其实转置卷积从几何上的理解，就是原本卷积的逆过程呗
# 之前的卷积都是通过一个卷积核，将输入的一片像素浓缩成一个很小的输出像素
# 这里的转置卷积就是反过来，利用卷积核对输入的一个像素做运算，放大成一片输出像素
# ********************************************************************************

# ========================
# 导入基础包
# ========================
import torch
from torch import nn
import d2lzh_pytorch as d2l

# ========================
# 基本操作
# ========================

# 简单的卷积核实现，不多解释
def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(trans_conv(X, K))

# 这里nn.ConvTranspose2d要求卷积核张量的形状为(1,1,2,2)，前面两个1是样本数和通道数的意思
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
# 顾名思义
tconv.weight.data = K
print(tconv(X))

# ========================
# 填充、步幅和多通道
# ========================

# 与常规卷积不同，在转置卷积中，填充被应用于的输出（常规卷积将填充应用于输入）
# 例如，当将高和宽两侧的填充数指定为1时，转置卷积的输出中将删除第一和最后的行与列

# 其实这是很合理的，因为在之前的理解中，卷积会使得输出的宽高缩小，而填充是为了把输出的宽高变大
# 在这里转置卷积是把输出的宽高变大，填充就反过来把输出的宽高变小就好了

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print(tconv(X))

# 步幅：同样，之前的卷积是以更大的效率浓缩，现在是以更大的效率放大
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print(tconv(X))

# 多通道操作：这个更简单，就不讲了
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(tconv(conv(X)).shape == X.shape)

# ========================
# 与矩阵变换的联系
# ========================

# 北海，要多想。
# 用脑子想想就好，实在不行打个草稿，就不注释了
