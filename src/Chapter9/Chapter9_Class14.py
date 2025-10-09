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
    实战Kaggle比赛：狗的品种识别（ImageNet Dogs）
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
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# 如果使用Kaggle比赛的完整数据集，请将下面的变量更改为False
demo = False
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = '../../data/dog-breed-identification'

def reorg_dog_data(data_dir, valid_ratio):
    # 读取标签
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    # 将验证集从原始的训练集中拆分出来
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    # 在预测期间整理测试集，以方便读取
    d2l.reorg_test(data_dir)

batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)

# ========================
# 图像增广
# ========================

# 训练集transform
transform_train = torchvision.transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08～1之间，高宽比在3/4和4/3之间。
    # 然后缩放为 224×224
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    # 以 50% 概率水平翻转
    torchvision.transforms.RandomHorizontalFlip(),
    # 随机调整亮度、对比度、饱和度（每次变化范围 ±40%）
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # 把图片从 PIL / NumPy 转成 PyTorch Tensor，像素值缩放到 [0,1]
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道，使用 ImageNet 统计均值与方差
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

# 测试集transform
transform_test = torchvision.transforms.Compose([
    # 将图片最短边缩放到 256 像素，保持比例
    torchvision.transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    torchvision.transforms.CenterCrop(224),
    # 把图片从 PIL / NumPy 转成 PyTorch Tensor，像素值缩放到 [0,1]
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道，使用 ImageNet 统计均值与方差
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

# ========================
# 读取数据集
# ========================

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

# ========================
# 微调预训练模型
# ========================

from torchvision import models
from torch.hub import load_state_dict_from_url

def get_net(device):
    finetune_net = nn.Sequential()
    # 预模型参数
    weights = load_state_dict_from_url(
        'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        model_dir=r"..\..\data\pretrainedmodels"
    )
    # 完整的 ResNet34（包含原 fc 层，1000输出）
    finetune_net.features = models.resnet34()
    finetune_net.features.load_state_dict(weights)
    # 定义一个新的输出网络，共有120个输出类别
    # 接fc,relu,fc从1000-256-120
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(device)
    # 冻结features层参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

# def get_net(device):
#     finetune_net = nn.Sequential()
#     # 完整的 ResNet34（包含原 fc 层，1000输出）
#     finetune_net.features = torchvision.models.resnet34(pretrained=True)
#     # 定义一个新的输出网络，共有120个输出类别
#     # 接fc,relu,fc从1000-256-120
#     finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
#                                             nn.ReLU(),
#                                             nn.Linear(256, 120))
#     # 将模型参数分配给用于计算的CPU或GPU
#     finetune_net = finetune_net.to(device)
#     # 冻结features层参数
#     for param in finetune_net.features.parameters():
#         param.requires_grad = False
#     return finetune_net

loss = nn.CrossEntropyLoss(reduction='none')

# 验证/评估阶段的损失计算函数
# 用当前网络计算在验证集或测试集上的平均损失
def evaluate_loss(data_iter, net, device):
    was_training = net.training
    net.eval()                              # 关掉 BN 更新 & Dropout
    l_sum, n = 0.0, 0
    with torch.no_grad():                   # 关梯度
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l_sum += l.sum()
            n += y.numel()
    if was_training:
        net.train()                         # 恢复原模式
    return (l_sum / n).to('cpu')

# ========================
# 定义训练函数
# ========================

import matplotlib.pyplot as plt
import time

def train(net, train_iter, valid_iter, num_epochs, lr, wd, device, lr_period,
          lr_decay):
    # 单GPU
    net = net.to(device)
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)

    # 记录曲线
    train_losses, valid_losses = [], []

    # 计时
    total_start = time.time()

    # 为吞吐量累计样本数（按实际 seen 的样本数，避免 drop_last 造成偏差）
    total_seen = 0

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        running_count = 0

        for features, labels in train_iter:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            trainer.zero_grad(set_to_none=True)
            outputs = net(features)
            l = loss(outputs, labels)  # 假设 loss 返回 per-sample 或 batch 总损；下面统一用 .sum()
            if l.ndim > 0:
                l = l.sum()
            l.backward()
            trainer.step()

            bs = labels.shape[0]
            running_loss += l.item()
            running_count += bs
            total_seen += bs

        # 该 epoch 的平均训练损失
        epoch_train_loss = running_loss / max(1, running_count)
        train_losses.append(epoch_train_loss)

        # 验证损失（若提供）
        if valid_iter is not None:
            with torch.no_grad():
                valid_loss = evaluate_loss(valid_iter, net, device)  # 期望返回标量张量
                if torch.is_tensor(valid_loss):
                    valid_loss_val = valid_loss.detach().float().cpu().item()
                else:
                    valid_loss_val = float(valid_loss)
            valid_losses.append(valid_loss_val)
            print(f'epoch {epoch + 1}: train loss {epoch_train_loss:.4f}, valid loss {valid_loss_val:.4f}')
        else:
            valid_losses.append(None)
            print(f'epoch {epoch + 1}: train loss {epoch_train_loss:.4f}')

        # 学习率衰减
        scheduler.step()

        # 训练总耗时与吞吐量
    total_time = time.time() - total_start
    examples_per_sec = total_seen / max(1e-12, total_time)

    # ---- 画图 ----
    plt.figure(figsize=(8, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, label='train loss')
    if valid_iter is not None:
        # 过滤 None，或直接画（长度一致）
        plt.plot(epochs, [v for v in valid_losses], label='valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 最终打印
    tail = f"train loss {train_losses[-1]:.4f}"
    if valid_iter is not None:
        tail += f", valid loss {valid_losses[-1]:.4f}"
    print(tail + f'\n{examples_per_sec:.1f} examples/sec on {device}')

# ========================
# 训练和验证模型
# ========================

num_epochs, lr, wd = 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(device)
train(net, train_iter, valid_iter, num_epochs, lr, wd, device, lr_period,
      lr_decay)

# ========================
# 在 Kaggle 上对测试集进行分类并提交结果
# ========================

net = get_net(device)
train(net, train_valid_iter, None, num_epochs, lr, wd, device, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(device)), dim=1)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
