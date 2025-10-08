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
    实战 Kaggle 比赛：图像分类 (CIFAR-10)
"""

# ========================
# 包和模块
# ========================
import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
import d2lzh_pytorch as d2l

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ========================
# 获取并组织数据集
# ========================
#@save
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# 如果使用完整的Kaggle竞赛的数据集，设置demo为False
demo = False

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../../data/cifar-10'

#@save
def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# 训练样本 :', len(labels))
print('# 类别 :', len(set(labels.values())))

#@save
def copyfile(filename, target_dir):
    """将文件复制到目标目录，如果已存在则跳过"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, os.path.basename(filename))
    if not os.path.exists(target_path):   # 如果不存在再复制
        shutil.copy(filename, target_path)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # ********************************************************************************
    # 最终目录结构形如：
    # train_valid_test/
    #     train_valid/
    #         cat/
    #         dog/
    #     train/
    #         cat/
    #         dog/
    #     valid/
    #         cat/
    #         dog/
    # ********************************************************************************
    # 训练数据集中样本最少的类别中的样本数
    # collections.Counter(labels.values()) 会统计每个类别的样本数量；
    # .most_common() 按数量从多到少排序
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数，即计算每类验证样本的数量
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    # 用一个字典 label_count 记录每个类别目前已分到验证集的数量
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        # 用 train_file.split('.')[0] 去掉文件后缀，比如 cat1.jpg → cat1
        # 再查 labels 获取类别名
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        # 首先把样本复制到一个“汇总文件夹” train_valid/label/ 下
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            # 取出键的值，如果没有这个键就是0
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label

#@save
def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    # ********************************************************************************
    # 执行完函数后，数据集结构变成：
    # train_valid_test/
    #     train/
    #         cat/
    #         dog/
    #     valid/
    #         cat/
    #         dog/
    #     test/
    #         unknown/
    #             img1.jpg
    #             img2.jpg
    #             ...
    # ********************************************************************************
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))

def reorg_cifar10_data(data_dir, valid_ratio):
    # 读取标签
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    # 将验证集从原始的训练集中拆分出来
    reorg_train_valid(data_dir, labels, valid_ratio)
    # 在预测期间整理测试集，以方便读取
    reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)

# ========================
# 图像增广
# ========================

# 使用图像增广来解决过拟合的问题

# 训练集的变换
transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    torchvision.transforms.Resize(40),
    # 从 40×40 图像中随机裁剪一个区域，裁剪区域的面积占原图的 64%~100%
    # 保持宽高比为 1（即正方形），
    # 然后将其缩放为高度和宽度均为32像素的正方形
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    # 以 50% 概率水平翻转图片
    torchvision.transforms.RandomHorizontalFlip(),
    # 把 PIL.Image 或 numpy.ndarray 转换成 PyTorch 的张量，
    # 并将像素值从 [0, 255] 缩放到 [0, 1]
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# 测试集不做随即变换，但转张量和标准化必须相同
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# ========================
# 读取数据集
# ========================

# ********************************************************************************
# torchvision.datasets.ImageFolder：
# ImageFolder 是 PyTorch 提供的最常用数据集类，它会根据文件夹结构自动为图片打标签。
# 要求结构如下：
# train_valid_test/
#     train/
#         cat/
#         dog/
#     train_valid/
#         cat/
#         dog/
# ********************************************************************************

# transform使用上面提到的图像增广
# 上述代码将数据集整理为对应格式后应用ImageFolder
# torchvision.datasets.ImageFolder会把原先的数据集加载成满足torch.utils.data.DataLoader的格式
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]

train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)

# ********************************************************************************
# 最终数据结构：
# train_valid_iter: 汇总原始训练集全部样本, 50000
# train_iter: 每类 4,500 × 10 类, 45000
# valid_iter: 每类 500 × 10 类, 5000
# test_iter: Kaggle 测试集（无标签）, 300,000
# ********************************************************************************

# ========================
# 定义Resnet-18模型
# ========================

def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")

# ========================
# 定义训练函数
# ========================

import matplotlib.pyplot as plt
import time

def train(net, train_iter, valid_iter, num_epochs, lr, wd, device, lr_period,
          lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    # 训练过程中，每隔 lr_period 个 epoch 就把学习率乘上一个衰减系数 lr_decay
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)

    # 用列表记录训练过程
    train_losses, train_accs, valid_accs = [], [], []

    # 用于统计总时间
    total_start = time.time()

    net = net.to(device)

    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            # l,acc分别是这个batch的train_loss_sum, train_acc_sum
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, device)
            metric.add(l, acc, labels.shape[0])

        # 计算并存储训练指标
        epoch_loss = metric[0] / metric[2]
        epoch_acc = metric[1] / metric[2]
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            valid_accs.append(valid_acc)
        else:
            valid_accs.append(None)

        scheduler.step()

        print(f'epoch {epoch+1}: train loss {epoch_loss:.3f}, '
              f'train acc {epoch_acc:.3f}, '
              f'valid acc {valid_accs[-1]:.3f}' if valid_iter else '')

    # ---- 画图 ----
    plt.figure(figsize=(8,6))
    epochs = range(1, num_epochs+1)
    plt.plot(epochs, train_losses, label='train loss')
    plt.plot(epochs, train_accs, label='train acc')
    if valid_iter is not None:
        plt.plot(epochs, valid_accs, label='valid acc')

    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 总耗时
    total_time = time.time() - total_start
    examples_per_sec = metric[2] * num_epochs / total_time

    measures = (f'train loss {train_losses[-1]:.3f}, '
                f'train acc {train_accs[-1]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_accs[-1]:.3f}'
    print(measures + f'\n{examples_per_sec:.1f} examples/sec on {str(device)}')

# ========================
# 训练和验证模型
# ========================

num_epochs, lr, wd = 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, device, lr_period,
      lr_decay)

# ========================
# 在 Kaggle 上对测试集进行分类并提交结果
# ========================

net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, device, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.to(device))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)

