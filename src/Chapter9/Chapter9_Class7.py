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
    单发多框检测（SSD）
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_computer-vision/ssd.html
# ********************************************************************************

# ********************************************************************************
# 😵这个版本训练一趟起码10小时，别等了，我有Plus版
# 😉直接大改特改，连绘图都改了，要看也别看这个
# ********************************************************************************

# ========================
# 类别预测层
# ========================

# ********************************************************************************
# 大概意思就是：
# 例如对概率预测：
#
# 1.卷积输出：
# 将特征图通过1*1或3*3卷积，得到一张 [a(q+1), h, w] 的特征图；
# a：每个像素位置对应多少个锚框；q + 1：每个锚框要预测的类别数（含背景）
#
# 2. 通道分块：
# 把输出通道按锚框拆分成 a 组，每组有 (q+1) 个通道。
# 对于第 k 个锚框，取出第 k 组通道 ⇒ 形状是 [q+1, h, w]
#
# 3.空间位置对应：
# 对于输入特征图上的每个位置 (i, j)
# 第 k 组的 (i, j) 处是一组长度 q+1 的向量。
# 这个向量就是 “以 (i, j) 为中心生成的第 k 个锚框” 对 q+1 类别 的预测分数
#
# 4.后续处理：
# 对这 q+1 个值做 softmax（或sigmoid，看具体实现）得到每类的概率。
# 取最大概率的类别作为预测类别，也可以直接输出整组概率用于计算损失。
# 回归偏移量（bounding box regression） 通常由另一条卷积头负责，
# 它的输出通道数是 a × 4（x、y、w、h 或四个坐标），结构完全类似，只是预测的是位置参数。
# ********************************************************************************

# ********************************************************************************
# 全连接参数量：
# 全连接输入维度：h × w × C
# 全连接输出参数：(h × w × a × (q + 1))
# 总参数：(h × w × C) × (h × w × a × (q+1)) + h × w × a × (q+1)
# 卷积参数量：
# 输入通道数：C
# 输出通道数：a × (q+1)
# 总参数：k × k × C × a × (q+1) + a × (q+1)
# 对比：
# 假设k = 3，C = 256，h = w = 38，a = 3，q = 20
# 则参数量分别是3.37×10¹⁰和1.46×10⁵
# ********************************************************************************

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import d2lzh_pytorch as d2l
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("当前 device:", device)
print("CUDA 可用:", torch.cuda.is_available())

print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

import time
from tqdm import tqdm   # 进度条库，pip install tqdm

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)

# ========================
# 边界框预测层
# ========================

def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

# ========================
# 连结多尺度的预测
# ========================

def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
print(Y1.shape)
print(Y2.shape)

def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

print(concat_preds([Y1, Y2]).shape)

# ========================
# 高和宽减半块
# ========================

def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)

# ========================
# 基本网络块
# ========================

def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)

# ========================
# 完整的模型
# ========================

def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    # 这里的旧版multibox_prior我换成了新版的MultiBoxPrior_torch，这俩输入输出是完全一致的
    anchors = d2l.MultiBoxPrior_torch(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)

# ========================
# 读取数据集和初始化
# ========================

batch_size = 256
train_iter, _ = d2l.load_data_bananas(batch_size)

net = TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# ========================
# 定义损失函数和评价函数
# ========================

cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

# ========================
# 训练模型
# ========================

num_epochs= 20
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
global_start = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    metric = d2l.Accumulator(4)
    net.train()

    # 记录上一个 batch 结束时间，用于计算数据加载耗时
    last_end = epoch_start

    for features, target in tqdm(train_iter,
                                 desc=f"Epoch {epoch + 1}/{num_epochs}",
                                 leave=False):
        t0 = time.time()
        data_load_time = t0 - last_end           # 仅数据加载时间

        # --- 数据搬运到 GPU ---
        t1 = time.time()
        X = features.to(device, non_blocking=True)
        Y = target.to(device, non_blocking=True)
        to_gpu_time = time.time() - t1

        # --- 前向、反向、更新 ---
        t2 = time.time()
        trainer.zero_grad()
        anchors, cls_preds, bbox_preds = net(X)
        bbox_labels, bbox_masks, cls_labels = d2l.MultiBoxTarget(
            anchors.to(device), Y.to(device))
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        l.mean().backward()
        trainer.step()
        train_time = time.time() - t2

        # 累加指标
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())

        # 打印各阶段耗时
        print(f"  load:{data_load_time:.3f}s | toGPU:{to_gpu_time:.3f}s | "
              f"train:{train_time:.3f}s | total:{time.time()-t0:.3f}s")

        last_end = time.time()   # 下个 batch 计算数据加载时间的基准

    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
    print(f"Epoch {epoch + 1} 完成, 用时 {time.time() - epoch_start:.2f} 秒 "
          f"(cls_err {cls_err:.3f}, bbox_mae {bbox_mae:.3f})")

total_time = time.time() - global_start
print(f"\n训练总耗时: {total_time:.2f} 秒, 平均 {total_time / num_epochs:.2f} 秒/epoch")
