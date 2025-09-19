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

# ********************************************************************************
# 输入形状是nh×nw，卷积核窗口形状是kh×kw，
# 在高的两侧一共填充ph行，在宽的两侧一共填充pw列，那么输出形状将会是
# (nh−kh+ph+1)×(nw−kw+pw+1)
# 如果希望输入和输出形状不变，填充一般设置为：
# ph=kh−1和pw=kw−1
# 当高上步幅为sh，宽上步幅为sw时，输出形状为：
# [(nh−kh+ph+sh)/sh⌋×⌊(nw−kw+pw+sw)/sw]
# ********************************************************************************

"""
    填充和步幅
"""

import torch
from torch import nn

# =================
# 填充
# =================

# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    # 元组拼接，结果变成 (1, 1, h, w)
    # 卷积神经网络里，通常期望输入张量的形状是四维：(batch_size, channels, height, width)
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])  # 排除不关心的前两维：批量和通道

# 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
# padding=1 → 在四周各补 1 行/列，所以卷积后高宽不变
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

X = torch.rand(8, 8)
print(comp_conv2d(conv2d, X).shape)

# 当卷积核的高和宽不同时，我们也可以通过设置高和宽上不同的填充数使输出和输入具有相同的高和宽
# 使用高为5、宽为3的卷积核。在高和宽两侧的填充数分别为2和1
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)

# =================
# 步幅
# =================

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)
