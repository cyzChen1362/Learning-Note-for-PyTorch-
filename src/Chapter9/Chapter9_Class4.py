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
# 注意这个s缩放系数，相当于锚框和图片的大小比值，所以上述宽/高是合理的
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
    # 以及，这里当然是加法而不是惩罚（没错，就是在你认为的广播机制条件下）
    # 因为shifts的某一行是一个特征图上某个像素位置对应到原图的归一化中心点坐标 [cx, cy, cx, cy]
    # base_anchors的每一行则是某个归一化中心点的归一化偏移量
    # 总偏移后坐标 = （宽or高） * （归一化中心点 + 归一化偏移量）
    # 或者更形象来说，这里计算的偏移量是s√r和s/√r，而总偏移量是ws√r和hs/√r
    # 刚好总中心点是wcx和hcy，这里的shifts和base_anchors都提取出来了w和h项
    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))

    # 最后的输出坐标也是归一化的
    # 第二维的0~n+m-2是同一中心点坐标不同锚框，以此类推
    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)


X = torch.Tensor(1, 3, h, w)  # 构造输入数据
Y = MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape) # torch.Size([1, 2042040, 4])

# 将锚框变量y的形状变为（图像高，图像宽，以相同像素为中心的锚框个数，4）
# 其实就是reshape的双重逆变换，仔细想一想（虽然确实相当抽象）
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

# ========================
# 交并比
# ========================

# ********************************************************************************
# 交并比:
# 真实边界框∩锚框 / 真实边界框∪锚框
# 1表示两个边界框相等，0表示两个边界框无重合像素
# ********************************************************************************

# 以下函数已保存在d2lzh_pytorch包中方便以后使用
# 参考https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py#L356

# 交集面积计算函数
# 输入：
# set_1：形状 (n1, 4)，每一行是一个框 (xmin, ymin, xmax, ymax)
# set_2：形状 (n2, 4)，同样的格式
# 目标：
# 得到一个 (n1, n2) 的矩阵，第 (i, j) 元素是 set_1[i] 与 set_2[j] 的交集面积。
def compute_intersection(set_1, set_2):
    """
    计算anchor之间的交集
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # PyTorch auto-broadcasts singleton dimensions
    # set_1[:, :2] 取每个框左上角 (xmin, ymin)
    # unsqueeze(1) 把它扩展成 (n1,1,2)
    # set_2[:, :2].unsqueeze(0) 是 (1,n2,2)
    # 通过广播后 torch.max 取更靠右下的左上角——这是交集矩形的左上角，(n1, n2, 2)
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    # 同理，取右下角坐标的最小值，得到交集矩形的右下角。
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    # upper_bounds - lower_bounds 得到交集矩形的宽高
    # 如果两个框不重叠，差值可能为负，clamp(min=0) 保证宽高至少为 0
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

# 输出 (n1, n2)，每个元素是 IoU，也就是交并比
def compute_jaccard(set_1, set_2):
    """
    计算anchor之间的Jaccard系数(IoU)
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # Find intersections
    # # 交集面积计算函数
    intersection = compute_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets

    # Sn1
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    # Sn2
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    # 并集面积，这里没有整什么并集面积计算公式，而是这样：
    # 并集 = S1 + S2 - S1∩S2
    # 同样，unsqueeze + 自动广播，形成 (n1,n2)
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    # 每个元素就是两个框的交并比
    return intersection / union  # (n1, n2)

# ========================
# 标注训练集的锚框
# ========================

# 理论部分看书，书上讲得很清楚，不需要笔记
# 笔记还不如直接看书呢

# bbox_scale 形如 (w, h, w, h)，方便把 (xmin, ymin, xmax, ymax) 乘回实际像素坐标
bbox_scale = torch.tensor((w, h, w, h), dtype=torch.float32)
# 真实边界框；
# 第一列是类别 id：0 表示“dog”，1 表示“cat”。
# 其余四列 (xmin, ymin, xmax, ymax) 都是 相对比例。
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                            [1, 0.55, 0.2, 0.9, 0.88]])
# 锚框
# 五个候选框（锚框），同样是比例坐标。
# 没有类别，只有坐标。
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
# 绘制真实框
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
# 绘制锚框
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
d2l.plt.show()

# 下面实现MultiBoxTarget函数来为锚框标注类别和偏移量

# 以下函数已保存在d2lzh_pytorch包中方便以后使用
# 直接看内部注释，不难
def assign_anchor(bb, anchor, jaccard_threshold=0.5):
    """
    # 按照「9.4.1. 生成多个锚框」图9.3所讲为每个anchor分配真实的bb, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        bb: 真实边界框(bounding box), shape:（nb, 4）
        anchor: 待分配的anchor, shape:（na, 4）
        jaccard_threshold: 预先设定的阈值
    Returns:
        assigned_idx: shape: (na, ), 每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
    """
    na = anchor.shape[0]
    nb = bb.shape[0]
    # 交并比
    jaccard = compute_jaccard(anchor, bb).detach().cpu().numpy() # shape: (na, nb)
    # 分配索引初始化
    assigned_idx = np.ones(na) * -1  # 初始全为-1

    # 先为每个bb分配一个anchor(不要求满足jaccard_threshold)
    jaccard_cp = jaccard.copy()
    for j in range(nb):
        # 看一看该真实边界框对应的列
        # 选定这一列中交并比最大的是哪一行
        i = np.argmax(jaccard_cp[:, j])
        # 对第i个锚框给定分配索引为j，即对应第j个真实边界框
        assigned_idx[i] = j
        # 既然第i给锚框已经分配了，那么删掉就好
        jaccard_cp[i, :] = float("-inf") # 赋值为负无穷, 相当于去掉这一行

    # 处理还未被分配的anchor, 要求满足jaccard_threshold
    for i in range(na):
        if assigned_idx[i] == -1:
            j = np.argmax(jaccard[i, :])
            if jaccard[i, j] >= jaccard_threshold:
                assigned_idx[i] = j

    return torch.tensor(assigned_idx, dtype=torch.long)

def xy_to_cxcy(xy):
    """
    将(x_min, y_min, x_max, y_max)形式的anchor转换成(center_x, center_y, w, h)形式的.
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    Args:
        xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    Returns:
        bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h

def MultiBoxTarget(anchor, label):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        anchor: torch tensor, 输入的锚框, 一般是通过MultiBoxPrior生成, shape:（1，锚框总数，4）
        label: 真实标签, shape为(bn, 每张图片最多的真实锚框数, 5)
               第二维中，如果给定图片没有这么多锚框, 可以先用-1填充空白, 最后一维中的元素为[类别标签, 四个坐标值]
    Returns:
        列表, [bbox_offset, bbox_mask, cls_labels]
        bbox_offset: 每个锚框的标注偏移量，形状为(bn，锚框总数*4)
        bbox_mask: 形状同bbox_offset, 每个锚框的掩码, 一一对应上面的偏移量, 负类锚框(背景)对应的掩码均为0, 正类锚框的掩码均为1
        cls_labels: 每个锚框的标注类别, 其中0表示为背景, 形状为(bn，锚框总数)
    """
    assert len(anchor.shape) == 3 and len(label.shape) == 3
    # batch_size
    bn = label.shape[0]

    def MultiBoxTarget_one(anc, lab, eps=1e-6):
        """
        MultiBoxTarget函数的辅助函数, 处理batch中的一个
        Args:
            anc: shape of (锚框总数, 4)
            lab: shape of (真实锚框数, 5), 5代表[类别标签, 四个坐标值]
            eps: 一个极小值, 防止log0
        Returns:
            offset: (锚框总数*4, )
            bbox_mask: (锚框总数*4, ), 0代表背景, 1代表非背景
            cls_labels: (锚框总数, 4), 0代表背景
        """
        an = anc.shape[0]
        # assign_anchor函数，真实框与锚框匹配
        # 这个lab也就是上一层函数的label，也就是真实边界框
        # 返回每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
        assigned_idx = assign_anchor(lab[:, 1:], anc) # (锚框总数, )
        # 先得到 (N,1) 的 0/1 列向量，作为是否有匹配到真实边界框的掩码
        # repeat(1,4) → (N,4)，方便与 4 个偏移量一一对应，复制四列
        # 正样本锚框掩码为 1，背景为 0
        bbox_mask = ((assigned_idx >= 0).float().unsqueeze(-1)).repeat(1, 4) # (锚框总数, 4)

        # 类别标签，初始化为背景0
        cls_labels = torch.zeros(an, dtype=torch.long) # 0表示背景
        # 同样，初始化锚框对应的bb坐标
        assigned_bb = torch.zeros((an, 4), dtype=torch.float32) # 所有anchor对应的bb坐标
        for i in range(an):
            bb_idx = assigned_idx[i]
            if bb_idx >= 0: # 即非背景
                cls_labels[i] = lab[bb_idx, 0].long().item() + 1 # 注意要加一
                assigned_bb[i, :] = lab[bb_idx, 1:]

        # 锚框对应坐标ccwh版
        center_anc = xy_to_cxcy(anc) # (center_x, center_y, w, h)
        # 真实边界框对应坐标ccwh版
        center_assigned_bb = xy_to_cxcy(assigned_bb)

        # 这里的公式见书本
        # 当然，书本是写了四项，这里是x y 两项算一次，w h 两项算一次，所以只有两条式子
        # 这里算x y偏移量
        offset_xy = 10.0 * (center_assigned_bb[:, :2] - center_anc[:, :2]) / center_anc[:, 2:]
        # 同理，这里算w h偏移量，除以0.1标准差就是乘以10
        offset_wh = 5.0 * torch.log(eps + center_assigned_bb[:, 2:] / center_anc[:, 2:])
        # 拼起来；如果都没有分配锚框那就没有偏移量，掩码覆盖为0
        offset = torch.cat([offset_xy, offset_wh], dim = 1) * bbox_mask # (锚框总数, 4)

        # view(-1)：拉平成 (N*4,) 一维向量，方便后续损失计算
        return offset.view(-1), bbox_mask.view(-1), cls_labels

    # 注意，这里是一张图片有很多个锚框，很多个偏移量
    # 所以 batch_offset 最终是一个 长度bn的列表，其中每个元素是一张图的 (N*4,) 张量
    # 其余同理
    batch_offset = []
    batch_mask = []
    batch_cls_labels = []
    for b in range(bn):
        # offset：形状 (N*4,)这一张图 所有N个锚框×4个偏移值，拉平的一维张量。
        # bbox_mask：形状 (N*4,)，同上。
        # cls_labels：形状 (N,)，这一张图 N 个锚框的类别标签。
        offset, bbox_mask, cls_labels = MultiBoxTarget_one(anchor[0, :, :], label[b, :, :])

        batch_offset.append(offset)
        batch_mask.append(bbox_mask)
        batch_cls_labels.append(cls_labels)

    # 原先是一个Python list
    # 现在在新的第一维把这些小张量“堆叠”起来，形成(bn, N*4)的单个 torch.Tensor
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    cls_labels = torch.stack(batch_cls_labels)

    return [bbox_offset, bbox_mask, cls_labels]

# 为锚框和真实边界框添加样本维
# 然后使用上面的MultiBoxTarget
# 输入锚框，真实边界框，返回偏移，掩码，类别标签
labels = MultiBoxTarget(anchors.unsqueeze(dim=0),
                        ground_truth.unsqueeze(dim=0))
# 输出锚框标注的类别
print(labels[2])

# ========================
# 输出预测边界框
# ========================

# 理论见书，很清楚了
# 注：一般非极大值抑制是“按类别独立执行”的
# 但下面代码这里没有，仅仅是看这个框最有可能是什么而已

# 首先构造4个锚框，假设预测偏移量全为0
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0.0] * (4 * len(anchors)))
cls_probs = torch.tensor([[0., 0., 0., 0.,],  # 背景的预测概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率

# 在图像上打印预测边界框和它们的置信度
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
d2l.plt.show()

# 实现MultiBoxDetection函数来执行非极大值抑制

# 以下函数已保存在d2lzh_pytorch包中方便以后使用
from collections import namedtuple
Pred_BB_Info = namedtuple("Pred_BB_Info", ["index", "class_id", "confidence", "xyxy"])

def non_max_suppression(bb_info_list, nms_threshold = 0.5):
    """
    非极大抑制处理预测的边界框
    Args:
        bb_info_list: Pred_BB_Info的列表, 包含预测类别、置信度等信息
        nms_threshold: 阈值
    Returns:
        output: Pred_BB_Info的列表, 只保留过滤后的边界框信息
    """
    output = []
    # 先根据置信度从高到低排序
    sorted_bb_info_list = sorted(bb_info_list, key = lambda x: x.confidence, reverse=True)

    while len(sorted_bb_info_list) != 0:
        # 取出并删除第一个元素
        best = sorted_bb_info_list.pop(0)
        # 添加到输出列表中
        output.append(best)

        # 如果元素取完了就退出循环呗
        if len(sorted_bb_info_list) == 0:
            break

        # 将删除最高置信度预测边界框后的其他边界框拎出来形成新列表
        bb_xyxy = []
        for bb in sorted_bb_info_list:
            bb_xyxy.append(bb.xyxy)

        # 计算最高置信度预测边界框和其他边界框的交并比
        iou = compute_jaccard(torch.tensor([best.xyxy]),
                              torch.tensor(bb_xyxy))[0] # shape: (len(sorted_bb_info_list), )

        # 开始筛选小于交并比阈值的那些，交并比过高说明重叠了
        n = len(sorted_bb_info_list)
        sorted_bb_info_list = [sorted_bb_info_list[i] for i in range(n) if iou[i] <= nms_threshold]
    return output

def MultiBoxDetection(cls_prob, loc_pred, anchor, nms_threshold = 0.5):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        cls_prob: 经过softmax后得到的各个锚框的预测概率, shape:(bn, 预测总类别数+1, 锚框个数)
        loc_pred: 预测的各个锚框的偏移量, shape:(bn, 锚框个数*4)，回忆一下书本里的偏移量，就是4个参数
        anchor: MultiBoxPrior输出的默认锚框, shape: (1, 锚框个数, 4)
        nms_threshold: 非极大抑制中的阈值
    Returns:
        所有锚框的信息, shape: (bn, 锚框个数, 6)
        每个锚框信息由[class_id, confidence, xmin, ymin, xmax, ymax]表示
        class_id=-1 表示背景或在非极大值抑制中被移除了
    """
    assert len(cls_prob.shape) == 3 and len(loc_pred.shape) == 2 and len(anchor.shape) == 3
    bn = cls_prob.shape[0]

    def MultiBoxDetection_one(c_p, l_p, anc, nms_threshold = 0.5):
        """
        MultiBoxDetection的辅助函数, 处理batch中的一个
        Args:
            c_p: (预测总类别数+1, 锚框个数)，参见461行“cls_probs”
            l_p: (锚框个数*4, )，也就是这一张图片的偏移量，对于某锚框的偏移量是4个参数
            anc: (锚框个数, 4)，也就是这一张图片的所有锚框的四个坐标，参见458行“anchors”
            nms_threshold: 非极大抑制中的阈值
        Return:
            output: (锚框个数, 6)
        """
        pred_bb_num = c_p.shape[1]
        # 锚框 + 这个锚框相对于它的真实边界框的偏移量 = 预测边界框
        anc = (anc + l_p.view(pred_bb_num, 4)).detach().cpu().numpy() # 加上偏移量

        # torch.max分别对每一列的所有行找出一个最大值
        # 返回两个张量：一个是最大值，一个是最大值的索引
        confidence, class_id = torch.max(c_p, 0)
        confidence = confidence.detach().cpu().numpy()
        class_id = class_id.detach().cpu().numpy()

        # 执行后：
        # pred_bb_info
        # [
        #   Pred_BB_Info(index=0, class_id=2, confidence=0.85, xyxy=[0.1,0.2,0.3,0.4]),
        #   Pred_BB_Info(index=1, class_id=-1, confidence=0.05, xyxy=[...]),
        #   ...
        # ]

        pred_bb_info = [Pred_BB_Info(
                            index = i,                  # 当前锚框编号
                            class_id = class_id[i] - 1, # 正类label从0开始
                            confidence = confidence[i], # 最大类别的置信度
                            xyxy=[*anc[i]])             # 边界框坐标列表
                        for i in range(pred_bb_num)]

        # 正类的index
        # 对预测框做一次非极大值抑制 (NMS)，然后提取被保留下来的预测框的索引(index)组成一个列表
        obj_bb_idx = [bb.index for bb in non_max_suppression(pred_bb_info, nms_threshold)]

        output = []
        for bb in pred_bb_info:
            output.append([
                # 筛出保留预测边界框的index
                (bb.class_id if bb.index in obj_bb_idx else -1.0),
                bb.confidence,
                *bb.xyxy
            ])

        return torch.tensor(output) # shape: (锚框个数, 6)

    batch_output = []
    for b in range(bn):
        batch_output.append(MultiBoxDetection_one(cls_prob[b], loc_pred[b], anchor[0], nms_threshold))

    return torch.stack(batch_output)

# Args:
# cls_prob: 经过softmax后得到的各个锚框的预测概率, shape:(bn=1, 预测总类别数+1, 锚框个数)
# offset_preds: 预测的各个锚框的偏移量, shape:(bn=1, 锚框个数*4)，回忆一下书本里的偏移量，就是4个参数
# anchor: MultiBoxPrior输出的默认锚框, shape: (1, 锚框个数, 4)

# Return:
# 所有锚框的信息, shape: (bn=1, 锚框个数, 6)
# 每个锚框信息由[class_id, confidence, xmin, ymin, xmax, ymax]表示
# class_id=-1 表示背景或在非极大值抑制中被移除了

output = MultiBoxDetection(
    cls_probs.unsqueeze(dim=0), offset_preds.unsqueeze(dim=0),
    anchors.unsqueeze(dim=0), nms_threshold=0.5)
print(output)

fig = d2l.plt.imshow(img)
for i in output[0].detach().cpu().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)

d2l.plt.show()
