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

"""
    多输入通道和多输出通道
"""

# =================
# 多输入通道
# =================

import torch
from torch import nn
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

def corr2d_multi_in(X, K):
    # 沿着X和K的第0维（通道维）分别计算再相加
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res

X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print(corr2d_multi_in(X, K))

# =================
# 多输出通道
# =================

def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    # stack没有指定参数，所以是默认dim = 0
    # torch.stack 的本质就是 在指定维度上新开一维，然后把一堆张量拼进去。所以结果的维度数一定会比原来多 1
    # dim只是指定插入的维度在哪里而已
    # 而cat是沿着已有维度粘贴，stack是新开维度
    return torch.stack([corr2d_multi_in(X, k) for k in K])

K = torch.stack([K, K + 1, K + 2])
print(K.shape) # torch.Size([3, 2, 2, 2])

print(corr2d_multi_in_out(X, K))

# =================
# 1×1卷积层
# =================

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    # 把每个通道的二维像素拉直，变成一列长度 h*w 的向量，结果是 c_i 行
    X = X.view(c_i, h * w)
    # 类似于W*X的W
    K = K.view(c_o, c_i)
    # 也就是第三章的内容
    Y = torch.mm(K, X)  # 全连接层的矩阵乘法
    return Y.view(c_o, h, w)

X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

print((Y1 - Y2).norm().item() < 1e-6)
