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
    图像增广
"""

# ********************************************************************************
# 通过对训练图像做一系列随机改变
# 来产生相似但又不同的训练样本，从而扩大训练数据集的规模
# 同时，随机改变训练样本可以降低模型对某些属性的依赖，从而提高模型的泛化能力
# ********************************************************************************


import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

import sys
sys.path.append("../..")
import d2lzh_pytorch as d2l
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ========================
# 绘图函数和辅助函数
# ========================

d2l.set_figsize()
img = Image.open('../../data/img/cat1.jpg')
d2l.plt.imshow(img)
d2l.plt.show()

# 绘图函数
# 作用是把很多张独立的图片按照网格排版到同一张画布上
# 本函数已保存在d2lzh_pytorch包中方便以后使用
# scale 是用来控制整张图的放大倍数的参数
def show_images(imgs, num_rows, num_cols, scale=2):
    # 这个figsize的作用是指定整张画布（Figure）的大小
    figsize = (num_cols * scale, num_rows * scale)
    # axes[i][j]：对应第 i 行第 j 列的那一块子图区域
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            # 对于每一行，都有num_cols列
            # 那么第1行的第2个自然是imgs中的第“1*num_cols + 2”个
            axes[i][j].imshow(imgs[i * num_cols + j])
            # 把每个子图的坐标轴刻度和坐标线都隐藏掉，让图片看起来更干净
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    d2l.plt.show()
    return axes

# 对同一张图片 img 反复做数据增强（augmentation），然后把生成的多张增强后的图片排成网格显示出来
# aug：传入的图像增强函数或变换，显然具有概率随机性，而不是对同一张图片反复用完全一样的变换
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)


# ========================
# 翻转和裁剪
# ========================

# 一半概率的图像水平（左右）翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())

# 一半概率的图像垂直（上下）翻转
apply(img, torchvision.transforms.RandomVerticalFlip())

# 随机裁剪出一块面积为原面积10%∼100%的区域
# 宽和高之比随机取自0.5∼2
# 宽和高分别缩放到200像素
apply(img, torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2)))

# ========================
# 变化颜色
# ========================

# 亮度随机变化为原图亮度的50%（1−0.5）∼150%（1+0.5）
apply(img, torchvision.transforms.ColorJitter(brightness=0.5))

# 随机变化图像的色调
# 色调是什么？https://chatgpt.com/s/t_68ce16f318d08191a63c2fa19d0dd58a
apply(img, torchvision.transforms.ColorJitter(hue=0.5))

# 随机变化图像的对比度
apply(img, torchvision.transforms.ColorJitter(contrast=0.5))

# 同时设置随机变化图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

# ========================
# 叠加多个图像增广方法
# ========================

# 形状算法
shape_aug = torchvision.transforms.RandomResizedCrop(
    200, scale=(0.1, 1), ratio=(0.5, 2))
# 颜色算法
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# 用Compose实例将上面定义的多个图像增广方法叠加起来
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)

# ========================
# 数据集准备工作
# ========================

# 下载CIFAR-10数据集
all_imges = torchvision.datasets.CIFAR10(train=True, root="~/Datasets/CIFAR", download=True)
# all_imges的每一个元素都是(image, label)
# 展示前32张图像
show_images([all_imges[i][0] for i in range(32)], 4, 8, scale=0.8);

# 为了在预测时得到确定的结果，我们通常只将图像增广应用在训练样本上
# 而不在预测时使用含随机操作的图像增广

# 这里我们只使用最简单的随机左右翻转
# 使用ToTensor将小批量图像转成PyTorch需要的格式
# 即形状为(批量大小, 通道数, 高, 宽)、值域在0到1之间且类型为32位浮点数
flip_aug = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

# 当然这是没有变换的算法，只用了ToTensor
no_aug = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

# 定义一个辅助函数来方便读取图像并应用图像增广
# 有关DataLoader的详细介绍，可参考3.5节
num_workers = 0 if sys.platform.startswith('win32') else 4
def load_cifar10(is_train, augs, batch_size, root="~/Datasets/CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    # 这里用到的 DataLoader 就是 torch.utils.data.DataLoader
    # 之所以代码里直接写 DataLoader(...)，是因为在前面已经有过导入
    # shuffle是否打乱，如果是测试集就不打乱了
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

# ========================
# 使用图像增广训练模型
# ========================

# 本函数已保存在d2lzh_pytorch包中方便以后使用
# 这里在CIFAR-10数据集上训练5.11节（残差网络）中介绍的ResNet-18模型
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)

            y_hat = net(X)
            l = loss(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

def train_with_data_aug(train_augs, test_augs, lr=0.001):
    # batch_size用的是256，net用的是输出为10类的18层ResNet
    batch_size, net = 256, d2l.resnet18(10)
    # 优化器用的是Adam
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 损失函数定义为交叉熵
    loss = torch.nn.CrossEntropyLoss()
    # load_cifar10的实现看上面
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    # 将训练集迭代器，测试集迭代器，网络，损失函数，优化器，gpu/cpu，轮数传入train中
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)

train_with_data_aug(flip_aug, no_aug)

