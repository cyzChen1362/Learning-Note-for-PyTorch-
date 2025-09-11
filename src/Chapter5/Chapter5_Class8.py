# ********************************************************************************
# 关于"它可以看成全连接层，其中空间维度（高和宽）上的每个元素相当于样本，通道相当于特征"的理解：
# 例如一个1*1的卷积核，并且输入通道数和输出通道数相同；
# 那么这一个1*1卷积层将会有 in_channels 组卷积核，每组卷积核有 in_channels 个卷积核
# 这 in_channels 个卷积核和一个（wide,height）位置的元素进行相乘相加
# 而这一个元素具有 in_channels 个特征
# 这就和全连接层的那个WX的某行乘以某列并sum得到输出的其中一个特征一样了
# ********************************************************************************

"""
    网络中的网络（NiN）
"""

# ========================
# NiN块
# ========================

import time
import torch
from torch import nn, optim

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    # 一个卷积层加两个充当全连接层的1×1卷积层
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU())
    return blk

# ========================
# NiN模型
# ========================

# 已保存在d2lzh_pytorch
import torch.nn.functional as F
# 结尾的那个全局平均池化层
class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        # 毕竟X的0、1维是样本数和通道数
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

net = nn.Sequential(
    # 54*54
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    # 26*26
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 26*26
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    # 12*12
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 12*12
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    # 5*5
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 5*5
    nn.Dropout(0.5),
    # 标签类别数是10
    # 5*5
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    # 1*1
    GlobalAvgPool2d(),
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    d2l.FlattenLayer())

# 构建一个数据样本来查看每一层的输出形状
X = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)

# ========================
# 获取数据和训练模型
# ========================

batch_size = 128
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.002, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

