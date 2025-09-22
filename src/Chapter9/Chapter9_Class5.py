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
    多尺度目标检测
"""

# ********************************************************************************
# 如果以图像每个像素为中心都生成锚框，很容易生成过多锚框而造成计算量过大
# 一种简单的方法是在输入图像中均匀采样一小部分像素，并以采样的像素为中心生成锚框
# 另一种方法是，在不同尺度下，我们可以生成不同数量和不同大小的锚框
# 即多尺度生成锚框
# ********************************************************************************

# ========================
# 导入实验所需的包
# ========================

from PIL import Image
import numpy as np
import torch

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

img = Image.open('../../data/img/catdog.jpg')
w, h = img.size # (728, 561)

# ========================
# 定义display_anchors函数
# ========================

# ********************************************************************************
# “我们在5.1节（二维卷积层）中将卷积神经网络的二维数组输出称为特征图。
# 我们可以通过定义特征图的形状来确定任一图像上均匀采样的锚框中心。”
# 意思就是：特征图每个像素映射回原图就是一片感受野，而这个感受野的中心就是锚框的中心
# ********************************************************************************

d2l.set_figsize()

def display_anchors(fmap_w, fmap_h, s):
    # 这里先输入特征图的宽和高
    # 前两维的取值不影响输出结果(原书这里是(1, 10, fmap_w, fmap_h), 我认为错了)
    fmap = torch.zeros((1, 10, fmap_h, fmap_w), dtype=torch.float32)

    # 然后特征图的宽有fmap_w，那就归一化锚框中心点分配为fmap_w个
    # fmap_h同理
    # 然后将归一化锚框中心点坐标乘以原图的宽和高就是锚框中心点在原图的位置了
    # 平移所有锚框使均匀分布在图片上
    offset_x, offset_y = 1.0/fmap_w, 1.0/fmap_h

    # MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    #     按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成(xmin, ymin, xmax, ymax).
    #     https://zh.d2l.ai/chapter_computer-vision/anchor.html
    #     Args:
    #         feature_map: torch tensor, Shape: [N, C, H, W].
    #         sizes: List of sizes (0~1) of generated MultiBoxPriores.
    #         ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores.
    #     Returns:
    #         anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1
    #         也就是这里MultiBoxPrior返回的是，将fmap上每个点作为锚框的中心点，然后返回4个锚框角的归一化坐标值
    # 如果直接映射到原图，每个锚框会“贴边”
    # 然后再加0.5offset，也就是向右下偏移，如果偏移1offset又另一条边贴边了
    anchors = d2l.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5]) + \
        torch.tensor([offset_x/2, offset_y/2, offset_x/2, offset_y/2])

    # 这里的img，w和h是外面的
    bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
    # 取出第0个batch做乘法
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)

# n+m-1，这里每个中心点有三种锚框，是正确的
display_anchors(fmap_w=4, fmap_h=2, s=[0.15])
d2l.plt.show()

# 将特征图的高和宽分别减半，并用更大的锚框检测更大的目标
display_anchors(fmap_w=2, fmap_h=1, s=[0.4])
d2l.plt.show()

# 将特征图的宽进一步减半至1，并将锚框大小增至0.8
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
d2l.plt.show()

# ********************************************************************************
# 这里直接说清楚目标检测/目标分割的大概思路
# 假设你已经有了MLP CNN 锚框这些思路
# 首先是总所周知的卷积层，然后会生成特征图；
# 特征图的每个像素都会对应一个锚框；
# 同时特征图的每个像素都会经过一个所谓的“检测头”，得到这个像素的“分类和偏移量”的预测值
# 记得这个像素对应的锚框吗？它会回到原图当中，然后去框原图（也就是锚框和真实框匹配）
# 然后框原图框出这个框对应的“分类和偏移量”的真实值
# 然后这个就是特征图的这个像素的“分类和偏移量”的真实值了，然后就去和预测值作比较
# 然后就是损失函数，反向传播，修改卷积层参数
# 所以实际上要做的事情就是：
# 训练一个卷积模型和检测头，让它生成的特征图经过“检测头”之后得出的，
# 每个像素的“分类和偏移量”是接近真实值的
# 反过来也就得到了在这个参数模型下，我塞进去图片，
# 预测出来图片的这一块区域的分类和偏移量是真实的
# ********************************************************************************
