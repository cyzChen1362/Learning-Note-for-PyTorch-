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
# 关于残差链接消除梯度消失的数学推导：https://chatgpt.com/share/68c6351b-3a78-8013-9133-0654b7f03315
# 关于反向传播的数学推导：https://cloud.tencent.com/developer/article/2037666
# ********************************************************************************

"""
    残差网络（ResNet）
"""

# ========================
# 残差块
# ========================

import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        # 3*3卷积块
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        # 3*3卷积块
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 如果使用残差，残差路径是一个1*1卷积层调整通道数
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        # 批归一化层，放在每个卷积输出后面
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        # 第一层卷积 + 批归一化 + 激活
        Y = F.relu(self.bn1(self.conv1(X)))
        # 第二层卷积 + 批归一化
        Y = self.bn2(self.conv2(Y))
        # 如果使用了残差那就旁边加一个，从输入越过两个卷积层直达输出
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

# 查看输入和输出形状一致的情况

blk = Residual(3, 3)
X = torch.rand((4, 3, 6, 6))
print(blk(X).shape) # torch.Size([4, 3, 6, 6])

# ========================
# ResNet模型
# ========================

# 首先是最开始进去的网络，和GoogLeNet进去的网络结构一致
net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 然后是选择使用残差或不使用残差，其中包含两个卷积块
# 可以参考论文的图
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

# 为ResNet加入所有残差块。这里每个模块使用两个残差块
# 每个resnet_blocki包含两个残差块，每个残差块有两层卷积
# 也就是一共十六层
net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))

# 与GoogLeNet一样，加入全局平均池化层后接上全连接层输出
net.add_module("global_avg_pool", d2l.GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10)))

# 加上最开始的卷积层和最后的全连接层，共计18层卷积层
# 也被称为ResNet-18

# 观察一下输入形状在ResNet不同模块之间的变化
X = torch.rand((1, 1, 224, 224))
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)

# ========================
# 获取数据和训练模型
# ========================
batch_size = 256
# 如出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

