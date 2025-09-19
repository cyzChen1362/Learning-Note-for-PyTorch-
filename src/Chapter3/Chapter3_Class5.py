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
torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
torchvision.utils: 其他的一些有用的方法。
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("../..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

"""
    图片分类数据集（Fashion-MNIST）
"""

# 下载Fashion-MNIST数据集，并分为训练集和测试集
# transform: 将原始的PIL图像（或者numpy数组）转换为PyTorch中使用的张量（Tensor）格式，并自动将像素值从[0, 255]缩放到[0.0, 1.0]之间。
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

# 输出训练集和测试集的长度
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))
# 通过下标访问任意一个样本
feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width

# 本函数已保存在d2lzh包中方便以后使用
# 只是对应label号返回名字而已
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
    # 启动SVG显示，更清晰
    d2l.use_svg_display()
    # 创建子图画布
    # 这里的_表示我们忽略（不使用）的变量
    # figs是一组子图（Axes）对象组成的数组或列表，用来在每个子图上绘制对应的图像
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    # 循环绘图
    # zip的作用：将三个可迭代对象 figs、images 和 labels 中的元素“打包”成一组组元组，一起进行遍历
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        # 不使用x和y坐标轴
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

X, y = [], []
for i in range(10):
    # mnist_train的第1维的0是图片，第1维的1是label
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 读取小批量图片
batch_size = 256
# 系统平台如果用的是win，则不用加速
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
# 用torch.utils.data.DataLoader方法，在mnist_train中读取batch_size批量图片
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# 用torch.utils.data.DataLoader方法，在mnist_test中读取batch_size批量图片
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 查看读取batch_size小批量图片所需时间
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))

