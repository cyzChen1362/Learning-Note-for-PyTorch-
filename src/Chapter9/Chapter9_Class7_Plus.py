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
    单发多框检测（SSD）
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_computer-vision/ssd.html
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

import matplotlib
# matplotlib.use("TkAgg")      # 或 "Qt5Agg"，任选系统支持的交互后端

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)


# ========================
# 手动画框验证
# ========================
# import torch
# import torchvision
# import matplotlib.pyplot as plt
# import d2lzh_pytorch as d2l
#
# # 读取图像
# X = torchvision.io.read_image('../../data/img/banana.jpg').unsqueeze(0).float()
# img = X.squeeze(0).permute(1, 2, 0).long()
#
# plt.figure(figsize=(5, 5), dpi=120)
# plt.imshow(img)
# h, w = img.shape[0:2]
#
# # 1️⃣ 归一化坐标
# manual_box = [0.1, 0.1, 0.8, 0.8]  # [xmin, ymin, xmax, ymax] (比例坐标)
#
# # 2️⃣ 转为像素坐标并转换为 Tensor
# manual_box_px = torch.tensor([
#     manual_box[0] * w,
#     manual_box[1] * h,
#     manual_box[2] * w,
#     manual_box[3] * h
# ])
#
# # 3️⃣ d2l.show_bboxes 需要一个二维张量或数组
# d2l.show_bboxes(plt.gca(),
#                 manual_box_px.unsqueeze(0),   # shape (1,4)
#                 labels=['test-box'],
#                 colors=['r'])
#
# plt.axis('off')
# plt.tight_layout()
# plt.show()
#

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

def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

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

# ========================
# 基本网络块
# ========================

def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

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

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter

num_epochs= 20
# animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
#                         ylim=None,
#                         legend=['class error', 'bbox mae'])
net = net.to(device)
global_start = time.time()
scaler = torch.cuda.amp.GradScaler()

# —— 新增：用列表保存训练过程数据
epochs_hist, cls_err_hist, bbox_mae_hist = [], [], []

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
        # trainer.zero_grad()
        # anchors, cls_preds, bbox_preds = net(X)
        # bbox_labels, bbox_masks, cls_labels = d2l.MultiBoxTarget(
        #     anchors.to(device), Y.to(device))
        # l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        # l.mean().backward()
        # trainer.step()
        # train_time = time.time() - t2
        trainer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = d2l.MultiBoxTarget_fast(anchors, Y)
            loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks).mean()
        scaler.scale(loss).backward()
        scaler.step(trainer)
        scaler.update()
        train_time = time.time() - t2

        # 累加指标
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())

        # 打印各阶段耗时
        print(f"  load:{data_load_time:.3f}s | toGPU:{to_gpu_time:.3f}s | "
              f"train:{train_time:.3f}s | total:{time.time()-t0:.3f}s")
        last_end = time.time()   # 下个 batch 计算数据加载时间的基准

    # —— 记录指标
    cls_err = 1 - metric[0] / metric[1]
    bbox_mae = metric[2] / metric[3]
    epochs_hist.append(epoch + 1)
    cls_err_hist.append(cls_err)
    bbox_mae_hist.append(bbox_mae)

    # cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    # animator.add(epoch + 1, (cls_err, bbox_mae))
    print(f"Epoch {epoch + 1} 完成, 用时 {time.time() - epoch_start:.2f} 秒 "
          f"(cls_err {cls_err:.3f}, bbox_mae {bbox_mae:.3f})")

total_time = time.time() - global_start
print(f"\n训练总耗时: {total_time:.2f} 秒, 平均 {total_time / num_epochs:.2f} 秒/epoch")

# —— 绘制 log y 轴曲线
fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
ax.plot(epochs_hist, cls_err_hist, label='class error', linewidth=1.8)
ax.plot(epochs_hist, bbox_mae_hist, label='bbox mae', linestyle='--', linewidth=1.8)
ax.set_xlabel('epoch')
ax.set_ylabel('error (log scale)')
ax.set_yscale('log')
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=None))     # 仅 10^n
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))       # 去掉次刻度
ax.yaxis.set_major_formatter(LogFormatter(base=10.0))            # 标签 10^n

ax.grid(True, which='major', linestyle=':')  # 只画主网格
ax.legend()
plt.tight_layout()
plt.show()

# ========================
# 预测目标（原书）
# ========================

X = torchvision.io.read_image('../../data/img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
print(output)
# def display(img, output, threshold):
#     d2l.set_figsize((5, 5))
#     fig = d2l.plt.imshow(img)
#     for row in output:
#         score = float(row[1])
#         if score < threshold:
#             continue
#         h, w = img.shape[0:2]
#         bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
#         d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
#
# display(img, output.cpu(), threshold=0.9)
# d2l.plt.show()

plt.figure(figsize=(5, 5), dpi=120)
plt.imshow(img)
h, w = img.shape[0:2]
for row in output:
    score = float(row[1])
    if score < 0.3:
        continue
    # 保证两个张量在同一设备
    scale = torch.tensor((w, h, w, h), device=row.device)
    # 先 clamp 到 [0,1] 再缩放到像素范围，避免负数或超出边界
    bbox = (row[2:6].clamp(0, 1) * scale).cpu()
    d2l.show_bboxes(plt.gca(), [bbox], f'{score:.2f}', 'w')
plt.axis('off')
plt.tight_layout()
plt.show()

# ========================
# 预测目标（train）
# ========================

X = torchvision.io.read_image('../../data/img/banana_train.png').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
print(output)
# def display(img, output, threshold):
#     d2l.set_figsize((5, 5))
#     fig = d2l.plt.imshow(img)
#     for row in output:
#         score = float(row[1])
#         if score < threshold:
#             continue
#         h, w = img.shape[0:2]
#         bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
#         d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
#
# display(img, output.cpu(), threshold=0.9)
# d2l.plt.show()

plt.figure(figsize=(5, 5), dpi=120)
plt.imshow(img)
h, w = img.shape[0:2]
for row in output:
    score = float(row[1])
    if score < 0.9:
        continue
    # 保证两个张量在同一设备
    scale = torch.tensor((w, h, w, h), device=row.device)
    # 先 clamp 到 [0,1] 再缩放到像素范围，避免负数或超出边界
    bbox = (row[2:6].clamp(0, 1) * scale).cpu()
    d2l.show_bboxes(plt.gca(), [bbox], f'{score:.2f}', 'w')
plt.axis('off')
plt.tight_layout()
plt.show()