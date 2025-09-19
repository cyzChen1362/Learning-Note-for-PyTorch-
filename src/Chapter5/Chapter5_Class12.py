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
# 关于稠密连接网络的数学推导：https://chatgpt.com/share/68c6351b-3a78-8013-9133-0654b7f03315
# ********************************************************************************

"""
    稠密连接网络（DenseNet）
"""

# ========================
# 稠密块
# ========================

import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# DenseNet使用了ResNet改良版的“批量归一化、激活和卷积”结构
# 这里每个卷积块都包含了批归一化+激活+卷积
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        # nn.Conv2d也不考虑输入的宽和高，不用担心这个参数
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        # 给出net的模块列表
        for i in range(num_convs):
            # 一层层下去通道数必然叠加
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        # 将模块列表进行注册，只注册模块和参数
        # 这些子卷积模块的参数将会出现在model.parameters() 中，方便优化器更新
        self.net = nn.ModuleList(net)
        # 上一级的输出拼接到下一级的输出
        # 例如上一级的输出通道数是 in_channels + (n - 1) * out_channels
        # 然后上一级的输出作为下一级的输入，而下一级的原始输出通道数是 out_channels
        # 然后跟上一级的输出通道数再拼接，就是 in_channels + (n - 1 + 1) * out_channels
        self.out_channels = in_channels + num_convs * out_channels # 计算输出通道数

    def forward(self, X):
        # self.net是ModuleList，继承自 Python 的 list
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X

# 定义一个有2个输出通道数为10的卷积块
# 最终是23通道的输出
blk = DenseBlock(2, 3, 10)
X = torch.rand(4, 3, 8, 8)
Y = blk(X)
print(Y.shape) # torch.Size([4, 23, 8, 8])

# ========================
# 过渡层
# ========================

# 用来控制模型复杂度
# 通过1×1卷积层来减小通道数，并使用步幅为2的平均池化层减半高和宽
# 进一步降低模型复杂度
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))
    return blk

# 将上面例子中的输出通道数由23降为10
blk = transition_block(23, 10)
print(blk(Y).shape) # torch.Size([4, 10, 4, 4])

# ========================
# DenseNet模型
# ========================

# 开头和ResNet一样的单卷积层和最大池化层
net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 和ResNet一样，使用4个稠密块，每块里面有4个卷积层，
# 最开始一个卷积层，最后一个全连接层，一共18层
# growing_rate即稠密块内每个卷积层的通道增长数
# 也就是一个稠密块的通道增加数为 growing_rate * num_convs
num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    # 这里一开始是先传入一个初始化为64的num_channels
    DB = DenseBlock(num_convs, num_channels, growth_rate)
    # 直接按次序往后添加模块
    net.add_module("DenseBlosk_%d" % i, DB)
    # 上一个稠密块的输出通道数
    # 这里就会在每一层更新num_channels了
    num_channels = DB.out_channels
    # 在最后一个稠密块之前
    # 在稠密块之间加入通道数减半的过渡层
    # 随着稠密块数量的增加，最后输出的通道数将会趋近于255
    # 当然也不能一直加，因为transition_block同样会使宽和高减半
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

# 同ResNet一样，最后接上全局池化层和全连接层来输出
net.add_module("BN", nn.BatchNorm2d(num_channels))
net.add_module("relu", nn.ReLU())
net.add_module("global_avg_pool", d2l.GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(num_channels, 10)))

# 打印每个子模块的输出维度
X = torch.rand((1, 1, 96, 96))
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)

# ========================
# 获取数据并训练模型
# ========================

batch_size = 256
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
