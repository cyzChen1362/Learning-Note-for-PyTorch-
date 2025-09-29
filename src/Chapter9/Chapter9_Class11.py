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
    全卷积网络
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_computer-vision/fcn.html
# ********************************************************************************

# ========================
# 导入基础包
# ========================
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import d2lzh_pytorch as d2l
import time

# ========================
# 构造模型
# ========================

# 具体模型见原书

# 首先使用的是ResNet-18中大部分的预训练层
# 输入320*480*3，输出512*10*15

# 这里使用和Chapter9_Class2同样的方法
# 预模型参数下载到指定地址 ..\..\data\pretrainedmodels
# 同时显式写出url

# pretrained_net = torchvision.models.resnet18(pretrained=True)

from torch.hub import load_state_dict_from_url
# 预模型参数
weights = load_state_dict_from_url(
    'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    model_dir=r"..\..\data\pretrainedmodels"
)

# pretrained_net.children()：
# 返回一个可迭代对象，里面是 pretrained_net 最外层的子模块
# [-3:]：模型结构中倒数第三、倒数第二和最后一个子模块

# 模型用resnet18
pretrained_net = torchvision.models.resnet18()
# 为我的模型加载参数
pretrained_net.load_state_dict(weights)
# 打印查看倒数三层
# print(list(pretrained_net.children())[-3:])

# 创建一个全卷积网络net，复制ResNet-18中大部分的预训练层，除了最后的全局平均汇聚层和最接近输出的全连接层
net = nn.Sequential(*list(pretrained_net.children())[:-2])

# 给定高度为320和宽度为480的输入，net的前向传播将输入的高和宽减小至原来的1/32，即10和15
# X = torch.rand(size=(1, 3, 320, 480))
# print(net(X).shape)

# 使用1*1卷积层将输出通道数转换为Pascal VOC2012数据集的类数（21类）
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))

# 回忆一下：
# 假设输入形状是nh×nw，卷积核窗口形状是kh×kw；
# 在高的两侧一共填充ph行，在宽的两侧一共填充pw列；
# 高上步幅为sh，宽上步幅为sw
# 此时输出形状为：
# [(nh−kh+ph+sh)/sh]×[(nw−kw+pw+sw)/sw]

# 那转置卷积正好是这个的逆过程，也就是已知输出为10*15，让你设计填充/步幅/卷积核，倒回去等于输入320*480
# 显然：(320 - 64 + 16 * 2 + 32)/32 = 10 且 (480 - 64 + 16 * 2 + 32)/32 = 15
# 当然这里的转置卷积层的参数还没有设置，在下面设置
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))

# 这里就是21*320*480

# ========================
# 初始化转置卷积层
# ========================

# 双线性插值：https://chatgpt.com/s/t_68d795974fc08191af8fa173f2d142f8

# 双线性插值算子
# 为了做“纯上采样”，通常要求 in_channels = out_channels，而且是逐通道处理

# 这里举一个例子来解释：
# 例如想要的是一个3*3的权重kernel，那么中心的权重就是1，然后由中心向外，第一圈有权重，第二圈没有权重
# 那么权重的计算方法大概是1 - 距离/factor，很显然距离=2的时候权重为0，也就是factor=2
# 然后中心的位置也很显然，如果kernel_size=3，那么center是1（0, 1, 2）中间那个
# 如果kernel_size=4，那么center是2.5（0, 1, 2, 3）中间那个

# og = (
#     torch.arange(3).reshape(-1, 1),
#     torch.arange(3).reshape(1, -1)
# )
# 生成：
# (
#  tensor([[0],
#          [1],
#          [2]]),      # og[0] : 形状 (3, 1)
#  tensor([[0, 1, 2]]) # og[1] : 形状 (1, 3)
# )

# weight[range(in_channels), range(out_channels), :, :] = filt
# 结果：
# weight[0,0,:,:] = filt
# weight[1,1,:,:] = filt
# ...非in_channels=out_channels的话weight=0,保证不跨通道

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

# 构造一个将输入的高和宽放大2倍的转置卷积层，并将其卷积核用bilinear_kernel函数初始化
# conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
#                                 bias=False)
# conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));

# 读取图像X，将上采样的结果记作Y
# img = torchvision.transforms.ToTensor()(d2l.Image.open('../../data/img/catdog.jpg'))
# X = img.unsqueeze(0)
# Y = conv_trans(X)

# 为了打印图像，我们需要调整通道维的位置
# out_img = Y[0].permute(1, 2, 0).detach()
#
# # 打印图像
# d2l.set_figsize()
# print('input image shape:', img.permute(1, 2, 0).shape)
# d2l.plt.imshow(img.permute(1, 2, 0));
# print('output image shape:', out_img.shape)
# d2l.plt.imshow(out_img);
# d2l.plt.show()

# 好的，这里终于设置转置卷积层transpose_conv的参数了
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);

# ========================
# 读取数据集
# ========================

batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

# ========================
# 训练
# ========================

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd = 5, 0.001, 1e-3
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

# 这里选择把转置卷积层的双线性插值参数固定
for param in net.transpose_conv.parameters():
    param.requires_grad = False

def train_batch_ch13(net, X, y, loss, trainer, device):
    """用多GPU进行小批量训练

    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # 微调BERT中所需
        X = [x.to(device) for x in X]
    else:
        X = X.to(device)
    y = y.to(device)

    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()

    # 这里返回的loss是sum起来的，所以后面会有一个metric[0] / metric[2]
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               device):
    """用单一GPU进行模型训练
        因为老子没有多GPU

    Defined in :numref:`sec_image_augmentation`"""
    # 使用 time 记录总时长
    total_time = 0.0

    # 记录曲线
    epochs, train_loss_list, train_acc_list, test_acc_list = [], [], [], []

    # 直接把模型搬到单个 device
    net = net.to(device)

    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            start_t = time.perf_counter()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, device)
            metric.add(l, acc, labels.shape[0], labels.numel())
            total_time += time.perf_counter() - start_t

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter, device)

        # 记录本轮结果
        epochs.append(epoch + 1)
        train_loss_list.append(metric[0] / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_acc)

        print(f'Epoch {epoch + 1}: '
              f'loss {metric[0] / metric[2]:.3f}, '
              f'train acc {metric[1] / metric[3]:.3f}, '
              f'test acc {test_acc:.3f}')

    print(f'{metric[2] * num_epochs / total_time:.1f} examples/sec on {device}')

    # ===== 用 matplotlib 画图 =====
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, train_acc_list, label='Train Acc')
    plt.plot(epochs, test_acc_list, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Curve')
    plt.xlim(1, num_epochs)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, device)

# ========================
# 预测
# ========================

def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(device)).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=device)
    X = pred.long()
    return colormap[X, :]

voc_dir = r"D:\LearningDeepLearning\LearningNote_Dive-into-DL-PyTorch\data\VOC2012"
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]

# d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
d2l.show_images(imgs[::3] + imgs[2::3], 2, n, scale=2);
d2l.plt.show()
