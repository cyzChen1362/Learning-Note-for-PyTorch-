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
import numpy as np
import sys
sys.path.append("../..")
import d2lzh_pytorch as d2l

"""
    多层感知机的从零开始实现
"""

# =======================
# 1. 获取和读取数据
# =======================
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# =======================
# 2. 定义模型参数
# =======================
# 每张图片用784个元素的向量进行表示，输入个数num_inputs=784
# 每张图片最终有10种可能，输出个数num_outputs=10
# 隐藏单元个数为256，即隐藏层输入784，输出256
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 隐藏层和输出层，分别有一个权重矩阵和偏置矩阵
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

# 打开求梯度
params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

# =======================
# 3. 定义激活函数
# =======================
# ReLU函数，隐藏层做了 XWh + bh 的结果将会套一个ReLU
# 之后再输出给下一层
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

# =======================
# 4. 定义模型
# =======================
def net(X):
    # [batch,1,28,28] --> [batch,1*28*28]
    X = X.view((-1, num_inputs))
    # 第一层隐藏层输出为H
    # matmul 是 matrix multiplication（矩阵乘法）
    H = relu(torch.matmul(X, W1) + b1)
    # 最终输出层输出
    return torch.matmul(H, W2) + b2

# =======================
# 5. 定义损失函数
# =======================
# 同3.7，softmax + 交叉熵
loss = torch.nn.CrossEntropyLoss()

# =======================
# 6. 训练模型
# =======================

# **********************************************************************
# 在Pytorch中，torch.nn.CrossEntropyLoss()得出的loss是每个batch的平均；
# 这个loss会比原书的mxnet小很多；
# 而sgd随机梯度下降是loss求导乘以learning rate除以batch；
# loss下降batch倍，则梯度下降效果下降batch倍
# 为了保持原有的学习效果，就需要对学习率乘以batch
# **********************************************************************

# **********************************************************************
# 学习率太高，参数的梯度下降幅度就会过大反而偏离了最佳参数
# 学习率太小，参数的梯度下降幅度就会过小一直都到不了最佳参数
# **********************************************************************

num_epochs, lr = 5, 0.5 * batch_size
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

