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
from matplotlib.ticker import LogLocator, LogFormatter
import time
from tqdm import tqdm   # 进度条库，pip install tqdm

# ========================
# 类别预测层
# ========================

def cls_predictor(num_inputs, num_anchors, num_classes):
    """
    类别预测层函数
    Args:
        num_inputs: 输入特征图的通道数
        num_anchors * (num_classes + 1): a * (q + 1)，每像素锚框数*（预测类别+背景）
    Return:
        nn.Conv2d: 类别预测卷积层
    """
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)

# ========================
# 边界框预测层
# ========================

def bbox_predictor(num_inputs, num_anchors):
    """
    边界框预测层函数
    Args:
        num_inputs: 输入特征图的通道数
        num_anchors * (num_classes + 1): a * 4，每像素锚框数*4个坐标
    Return:
        nn.Conv2d: 类别预测卷积层
    """
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

# ========================
# 连结多尺度的预测
# ========================

def flatten_pred(pred):
    # 先将pred的维度从(N, C, H, W)变为(N, H, W, C)，
    # 把“空间位置”和“每个位置对应的所有锚框预测”绑在一起看成一个整体
    # 然后将pred的维度从(N, H, W, C)变为[N, H*W*C]
    # 其中，pred的C维是num_anchors × (num_classes + 1)，对应一张特征图能展开出来的通道
    # 没错这里的N维是batch_size
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    # 可以去到TinySSD类下看concat_preds的使用
    # preds传入长度为 5 的列表，每个元素是 [N, C, H, W]
    # 沿着特征维（dim=1）拼接，得到第二维就是所有特征层的预测合并后的总长度
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

# ========================
# 高和宽减半块
# ========================

def down_sample_blk(in_channels, out_channels):
    """
    高和宽减半块函数
    Blocks:
        Conv2d(仅通道调整) + BatchNorm2d(通道维正则化) + ReLu
        Conv2d(仅通道调整) + BatchNorm2d(通道维正则化) + ReLu
    Return:
        Blocks + MaxPool2d(2)
    """
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
    # 通道改变：3 —— 16 —— 32 —— 64
    num_filters = [3, 16, 32, 64]
    # 三个高和宽减半块串联，从256*256到32*32
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

# ========================
# 完整的模型
# ========================

def get_blk(i):
    """
        完整的单发多框检测模型
        Blocks:
            基本网络块(Channels：3 —— 64)
            高和宽减半块(Channels：64 —— 128)
            高和宽减半块(Channels：128 —— 128)
            高和宽减半块(Channels：128 —— 128)
            全局最大池(高度和宽度都降到1)
        Return:
            Blocks
        """
    if i == 0:
        # 基本网络块
        blk = base_net()
    elif i == 1:
        # 高和宽减半块
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    # 单发多框检测模型输出的特征图
    Y = blk(X)
    # 这里的旧版multibox_prior我换成了新版的MultiBoxPrior_GPU，这俩输入输出是完全一致的
    # 锚框生成函数
    anchors = d2l.MultiBoxPrior_GPU(Y, sizes=size, ratios=ratio)
    # 类别预测函数
    cls_preds = cls_predictor(Y)
    # 边界框预测函数
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

# setattr(obj, name, value)
# 给对象 obj 新增或修改一个属性，名字是字符串 name，值是 value
# getattr(obj, name)
# 按照字符串 name 读取对象 obj 的属性

# ********************************************************************************
# 在下面的代码中，可以清晰地看到get_blk(i)对应cls_predictor(i)
# 也就是说，模型每下去一个blk，对应的feature_map就会经过一次cls_predictor
# 这就很合理了，因为每下去一个blk，feature_map的高和宽就会缩小⭐
# 也就是说，每一层下去都会用到不同尺度的锚框去做预测⭐
# 不同尺度锚框的预测，就是SSD的多尺度检测思想的核心⭐
#
# 更进一步，cls_predictor和bbox_predictor输出特征图每个像素点的预测情况
# 特征图每个像素点的真实情况是这个像素点对应锚框和原图对应区域的交并比
# 即通过学习卷积层的权重，使得输出特征图每个像素点的预测情况去拟合对应位置的真实情况⭐
# ********************************************************************************

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
        # 提前创建[None, None, None, None, None]
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # It is clear that 每块前向传播之后预测一下，
            # 同时前向传播的结果也放到下一块的输入
            # 并记录这一块预测的结果
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

# ********************************************************************************
# 在上面的代码中，anchors、cls_preds、bbox_preds 的形状和它们之间如何对齐呢？
# anchors[i]：[1, H*W*a, 4]
# cls_preds[i]：[N, a*(num_classes+1), H, W]
# bbox_preds[i]：[N, a*4, H, W]
# Then：
# anchors：[1, Σ_i H_i*W_i*a, 4]
# cls_preds：concat_preds 先 permute → [N, H_i, W_i, a*(num_classes+1)]
#            再展平 → [N, H_i*W_i*a*(num_classes+1)]
#            5 层拼接 → [N, Σ_i H_i*W_i*a*(num_classes+1)]
#            reshape → [N, total_anchors, num_classes+1]（方便交叉熵）
# bbox_preds：[N, total_anchors * 4]
# ********************************************************************************

# ========================
# 读取数据集和初始化
# ========================

batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)

net = TinySSD(num_classes=1)
# 优化器 / 训练器
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

# ========================
# 定义损失函数和评价函数
# ========================

"""
    reduction取值
    type:
        mean(默认)：先按元素计算损失，再取平均值，得到一个标量
        sum：先按元素计算损失，再全部求和，得到一个标量
        none：不做任何汇总，直接返回逐样本 / 逐元素的损失张量
    """

# L1范数损失：预测值和真实值之差的绝对值；
# 不用平方损失是因为平方损失的梯度会随着偏差线性增长，梯度容易过大
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    """
        损失函数：
        Args:
            cls_preds:：分类预测值；
                       [N, total_anchors, num_classes+1] —— [N * total_anchors, num_classes+1]
            cls_labels：分类真实值；
                       [N, total_anchors] —— [N * total_anchors]
            bbox_preds：偏移量预测值：
                       [N, total_anchors * 4]
            bbox_labels：偏移量真实值：
                       [N, total_anchors * 4]
            bbox_masks：边框坐标掩码：
                       [N, total_anchors * 4]
        Returns:
            cls + bbox：总损失 = 平均分类损失（交叉熵） + 平均边框回归损失（L1损失）
        """
    # cls：交叉熵返回[N * total_anchors] —— [N , total_anchors] —— [N]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    # bbox：L1损失返回[N , 4] —— [N]
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

# 评价函数
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    # 经典操作，取最后一维，转类型并和label比较，torch.sum直接求和得到标量
    # return [1]
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    # return [1]
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

# ========================
# 训练模型
# ========================

num_epochs= 20
net = net.to(device)
global_start = time.time()
# 使用梯度缩放器，在16位浮点数的情况下可以防止梯度下溢0
# 所以可以放心使用F16，不固定使用32，提高运算速度
scaler = torch.cuda.amp.GradScaler()

# 用列表保存训练过程数据
epochs_hist, cls_err_hist, bbox_mae_hist = [], [], []

for epoch in range(num_epochs):
    epoch_start = time.time()

    # 生成一个长度为 4 的累加器 metric = [0.0, 0.0, 0.0, 0.0]
    # 后面每个batch循环的结果返回也是在这个累加器里面累加而已
    # cls_eval:            统计整个 batch 中预测类别与真实类别相同的锚框数量
    # cls_labels.numel():  当前 batch 的锚框总数（= N × total_anchors）
    # bbox_eval:           计算所有正样本锚框坐标的绝对误差总和
    # bbox_labels.numel(): 参与回归的坐标数量（= N × total_anchors × 4）
    # 最终结果：
    # metric[0]：所有 batch 的分类正确数总和
    # metric[1]：所有 batch 的锚框总数
    # metric[2]：所有 batch 的边框绝对误差总和
    # metric[3]：所有 batch 的坐标数量总和
    metric = d2l.Accumulator(4)

    net.train()

    # 记录上一个 batch 结束时间，用于计算数据加载耗时
    last_end = epoch_start
    for features, target in tqdm(train_iter,
                                 desc=f"Epoch {epoch + 1}/{num_epochs}",
                                 leave=False):
        # 数据加载时间
        t0 = time.time()
        data_load_time = t0 - last_end

        # 数据搬运到 GPU
        t1 = time.time()
        X = features.to(device, non_blocking=True)
        Y = target.to(device, non_blocking=True)
        # 数据搬运时间
        to_gpu_time = time.time() - t1

        # 前向、反向、更新
        t2 = time.time()
        trainer.zero_grad(set_to_none=False) # 不知道为什么，给了False结果就很好...
        # AMP 上下文管理器，开启自动混合精度
        # 这里不用担心SSD五层有五个输出之类的问题，前面五个输出已经全部人人平等为预测样本了
        with torch.cuda.amp.autocast():
            # 预测值由net得出
            anchors, cls_preds, bbox_preds = net(X)
            # 真实值由MultiBoxTarget_fast得出
            bbox_labels, bbox_masks, cls_labels = d2l.MultiBoxTarget_fast(anchors, Y)
            # 计算损失函数
            loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks).mean()
        # loss 先被 scaler.scale() 放大，backward() 会计算梯度，但这些梯度是放大后的
        scaler.scale(loss).backward()
        # 这里会先检查梯度有没有出现 NaN/Inf，如果正常，就把梯度缩放回去（除以 scale 倍数）
        # 如果发现梯度坏掉（NaN/Inf），就跳过这次参数更新，防止网络崩掉
        scaler.step(trainer)
        # 根据刚才的情况动态调整 scale 值，
        # 如果最近一段时间都没溢出 → scale 增大（提升数值精度）
        # 如果经常溢出 → scale 减小（保证稳定性）
        scaler.update()
        # 数据训练时间
        train_time = time.time() - t2

        # 累加指标
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())

        # 打印各阶段耗时
        print(f"  load:{data_load_time:.3f}s | toGPU:{to_gpu_time:.3f}s | "
              f"train:{train_time:.3f}s | total:{time.time()-t0:.3f}s")
        last_end = time.time()   # 下个 batch 计算数据加载时间的基准

    # 错误分类占比
    cls_err = 1 - metric[0] / metric[1]
    # 边框坐标平均绝对误差
    bbox_mae = metric[2] / metric[3]

    # 列表，方便后面画图
    epochs_hist.append(epoch + 1)
    cls_err_hist.append(cls_err)
    bbox_mae_hist.append(bbox_mae)

    print(f"Epoch {epoch + 1} 完成, 用时 {time.time() - epoch_start:.2f} 秒 "
          f"(cls_err {cls_err:.3f}, bbox_mae {bbox_mae:.3f})")

total_time = time.time() - global_start
print(f"\n训练总耗时: {total_time:.2f} 秒, 平均 {total_time / num_epochs:.2f} 秒/epoch")

# —— 绘制 log y 轴曲线
fig, ax = d2l.plt.subplots(figsize=(8, 5), dpi=120)
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
d2l.plt.tight_layout()
d2l.plt.show()

# ========================
# 预测目标（原书）
# ========================

X = torchvision.io.read_image('../../data/img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

def predict(X):
    # 评估模式，上面循环epoch训练的时候net的参数已经被记住了
    net.eval()
    # 调用net，返回锚框，类型预测值，偏移量预测值
    anchors, cls_preds, bbox_preds = net(X.to(device))
    # 对最后一维（类别维度）做 softmax，得到每个 anchor 的类别概率
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    # output 形状 (batch_size, num_anchors, 6)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    # 预测通常一次只处理一张图片，所以取 output[0]，其形状(num_anchors, 6)
    # i：该锚框在这一批图片里的索引
    # row：长度为 6 的张量，对应上面那六个值，row[0] 是 class_id
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    # output[0, idx] 的形状 (len(idx), 6)
    # 即当前图片中通过 NMS 和阈值筛选的有效目标框
    return output[0, idx]

output = predict(X)
print(output)

d2l.plt.figure(figsize=(5, 5), dpi=120)
d2l.plt.imshow(img)
h, w = img.shape[0:2]
for row in output:
    score = float(row[1])
    if score < 0.9:
        continue
    # 保证两个张量在同一设备
    scale = torch.tensor((w, h, w, h), device=row.device)
    # 先 clamp 到 [0,1] 再缩放到像素范围，避免负数或超出边界
    bbox = (row[2:6].clamp(0, 1) * scale).cpu()
    d2l.show_bboxes(d2l.plt.gca(), [bbox], f'{score:.2f}', 'w')
d2l.plt.axis('off')
d2l.plt.tight_layout()
d2l.plt.show()

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
d2l.plt.figure(figsize=(5, 5), dpi=120)
d2l.plt.imshow(img)
h, w = img.shape[0:2]
for row in output:
    score = float(row[1])
    if score < 0.9:
        continue
    # 保证两个张量在同一设备
    scale = torch.tensor((w, h, w, h), device=row.device)
    # 先 clamp 到 [0,1] 再缩放到像素范围，避免负数或超出边界
    bbox = (row[2:6].clamp(0, 1) * scale).cpu()
    d2l.show_bboxes(d2l.plt.gca(), [bbox], f'{score:.2f}', 'w')
d2l.plt.axis('off')
d2l.plt.tight_layout()
d2l.plt.show()