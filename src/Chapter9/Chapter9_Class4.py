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
    锚框
"""

# ********************************************************************************
# 以每个像素为中心生成多个大小和宽高比（aspect ratio）不同的边界框
# 这些边界框被称为锚框（anchor box）
# ********************************************************************************

# ========================
# 导入实验所需的包
# ========================

from PIL import Image
import numpy as np
import math
import torch

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
print(torch.__version__) # 我的是 2.7.0+cu128

# ========================
# 生成多个锚框
# ========================

# ********************************************************************************
# 假设输入图像高为h，宽为w。我别以图像的每个像素为中心生成不同形状的锚框。
# 设大小为s∈(0,1]且宽高比为r>0，那么锚框的宽和高将分别为ws√r和hs/√r
# 注意这个s时隔缩放系数，相当于锚框和图片的大小比值，所以上述宽/高是合理的
#
# 分别设定好一组大小s1,…,sn和一组宽高比r1,…,rm。
# 如果以每个像素为中心时使用所有的大小与宽高比的组合，输入图像将一共得到whnm个锚框
# 毕竟是每个像素点为中心生成一个锚框，所以是合理的
# 通常只对包含s1或r1的大小与宽高比的组合感兴趣
#  也就是说，以相同像素为中心的锚框的数量为n+m−1。
#  对于整个输入图像，我们将一共生成wh(n+m−1)个锚框
# ********************************************************************************

d2l.set_figsize()
img = Image.open('../../data/img/catdog.jpg')
w, h = img.size
print("w = %d, h = %d" % (w, h)) # w = 728, h = 561

# 锚框生成函数
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    r"""
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        feature_map: torch tensor, Shape: [N, C, H, W].
        sizes: List of sizes (0~1) of generated MultiBoxPriores.
        ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores.
    Returns:
        anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1
    """
    pairs = [] # pair of (size, sqrt(ration))
    # 只对包含s1或r1的大小与宽高比的组合感兴趣
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])

    # 将嵌套列表转为NumPy array，才有“所有行的第一列”
    # 总长度为n+m-1
    pairs = np.array(pairs)

    # 宽系数
    ss1 = pairs[:, 0] * pairs[:, 1] # size * sqrt(ration)
    # 高系数
    ss2 = pairs[:, 0] / pairs[:, 1] # size / sqrt(ration)

    # 生成形如 [xmin, ymin, xmax, ymax] 的二维数组，n+m-1行，4列
    # 这些坐标是相对于中心点的半宽半高，宽系数 0.5 → 左右坐标 ±0.25
    base_anchors = np.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2

    # 取出倒数两个维度的大小
    h, w = feature_map.shape[-2:]
    # 横向锚框中心点坐标(归一化到0~1)
    shifts_x = np.arange(0, w) / w
    # 纵向锚框中心点坐标(归一化到0~1)
    shifts_y = np.arange(0, h) / h

    # shift_x：每一行都复制 shifts_x，表示每个网格点的横坐标
    # [[0, 1/w, 2/w, ...],
    #  [0, 1/w, 2/w, ...],
    #  ...
    # ]
    # shift_y：每一列都复制 shifts_y，表示每个网格点的纵坐标
    # [[0, 0, 0, ...],
    #  [1/h, 1/h, 1/h, ...],
    #  ...
    # ]
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)

    # 拉平成一维
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    # 把 (x, y) 复制成 [x, y, x, y]
    # → 行
    # ↓ 列  [[0  , 1/w, 2/w, ... , 0  , 1/w, 2/w, ...],
    #        [0  , 0  , 0  , ... , 1/h, 1/h, 1/h, ...],
    #        [0  , 1/w, 2/w, ... , 0  , 1/w, 2/w, ...],
    #        [0  , 0  , 0  , ... , 1/h, 1/h, 1/h, ...],
    #       ]
    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

    # shifts.reshape：行数仍然是h*w，第三维仍然是4，新增了1个第二维
    # 也就是shifts.reshape是(h*w,1,4)，base_anchors.reshape是(1,n+m-1,4)
    # 就能广播机制了
    # 生成的anchors是(h*w,n+m-1,4)
    # anchors的每一行都对应着同一个中心坐标，
    # 这一行的每一列都代表着一种不同的锚框大小，
    # 第三维4个通道说明每个中心坐标的每种锚框都由四个坐标点构成
    # 当然以上全是归一化的形式
    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))

    # 最后的输出坐标也是归一化的
    # 第二维的0~n+m-2是同一中心点坐标不同锚框，以此类推
    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)


X = torch.Tensor(1, 3, h, w)  # 构造输入数据
Y = MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape) # torch.Size([1, 2042040, 4])

# 将锚框变量y的形状变为（图像高，图像宽，以相同像素为中心的锚框个数，4）
# 其实就是reshape的双重逆变换，仔细想一想
boxes = Y.reshape((h, w, 5, 4))
# 访问以（250，250）为中心的第一个锚框
# 第四维的坐标分别是[xmin, ymin, xmax, ymax]
# * torch.tensor([w, h, w, h], dtype=torch.float32)
print(boxes[250, 250, 0, :]) # tensor([-0.0316,  0.0706,  0.7184,  0.8206])

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def show_bboxes(axes, bboxes, labels=None, colors=None):
    # 把输入统一转成列表（list）或元组（tuple），方便后面按下标取
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    # 把 labels 统一成列表或 None
    labels = _make_list(labels)
    # 如果 colors 没给，就使用默认颜色列表 ['b','g','r','m','c']，然后也统一成列表
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])

    for i, bbox in enumerate(bboxes):
        # 从颜色列表中取一个，i % len(colors) 是循环取色
        color = colors[i % len(colors)]
        # 把 [xmin, ymin, xmax, ymax] 转为 matplotlib.patches.Rectangle 对象，并设置边框颜色
        rect = d2l.bbox_to_rect(bbox.detach().cpu().numpy(), color)
        # axes.add_patch(rect)：把矩形添加到绘图区
        axes.add_patch(rect)
        # 添加文字标签
        if labels and len(labels) > i:
            # 如果边框颜色是白色 'w'，文字用黑色 'k'；
            # 否则文字用白色，确保对比度。
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=6, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

d2l.set_figsize()
fig = d2l.plt.imshow(img)
bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
# 画出以250,250为中心的所有锚框
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.75, r=2', 's=0.55, r=0.5', 's=0.5, r=1', 's=0.25, r=1'])
d2l.plt.show()


