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
    微调-热狗识别
"""

# ********************************************************************************
# 首先我们在一个大的数据集上训练出一个好的模型
# 但我们实际需要的是一个小的数据集进行应用
# 如果将大数据集的模型直接迁移到小数据集上用的话，容易过拟合
# 所以可以将源模型借鉴过去
#
# 具体步骤：
# 1. 在源数据集（如ImageNet数据集）上预训练一个神经网络模型，即源模型。
# 2. 创建一个新的神经网络模型，即目标模型。它复制了源模型上除了输出层外的所有模型设计及其参数。
# 我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。
# 我们还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用。
# 3. 为目标模型添加一个输出大小为目标数据集类别个数的输出层，并随机初始化该层的模型参数。
# 4. 在目标数据集（如椅子数据集）上训练目标模型。
# 我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的。
# ********************************************************************************

# ========================
# 导入实验所需的包
# ========================

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ========================
# 获取数据集
# ========================

# 数据集路径
data_dir = '../../data/S1/CSCL/tangss/Datasets'
# 这里会返回一下hotdog下有什么目录
os.listdir(os.path.join(data_dir, "hotdog")) # ['train', 'test']

# from torchvision.datasets import ImageFolder
# 创建两个ImageFolder实例来分别读取训练数据集和测试数据集中的所有图像文件
train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))

# 数据集的形状：https://chatgpt.com/s/t_68ce4c3f82f08191ad847aab86c92c09
# 取正数据集的第1到第8张图片
hotdogs = [train_imgs[i][0] for i in range(8)]
# 取负数据集的倒数第1到第8张图片
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

# ********************************************************************************
# 预训练模型的参数是通过怎样预处理的数据训练出来，
# 我们使用的时候就需要对我们自己的数据做怎样的预处理
#
# 如果使用的是torchvision的models，要求：
# 尺寸：RGB，至少 224×224
# 像素值范围：先缩放到 [0,1]
# 再按 mean = [0.485, 0.456, 0.406]、std = [0.229, 0.224, 0.225] 做标准化
# ********************************************************************************

# 指定RGB三个通道的均值和方差来将图像通道归一化
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
        # 先从图像中裁剪出随机大小和随机高宽比的一块随机区域
        # 然后将该区域缩放为高和宽均为224像素的输入
        transforms.RandomResizedCrop(size=224),
        # 然后随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 转Tensor
        transforms.ToTensor(),
        # 标准化
        normalize
    ])

test_augs = transforms.Compose([
        # 高和宽均缩放为256像素
        transforms.Resize(size=256),
        # 从中裁剪出高和宽均为224像素的中心区域作为输入
        transforms.CenterCrop(size=224),
        # 转Tensor
        transforms.ToTensor(),
        # 标准化
        normalize
    ])

# ========================
# 定义和初始化模型
# ========================

# 这里我稍微改得和教程有点不一样
# 预模型参数下载到指定地址 ..\..\data\pretrainedmodels
# 同时显式写出url

from torch.hub import load_state_dict_from_url
# 预模型参数
weights = load_state_dict_from_url(
    'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    model_dir=r"..\..\data\pretrainedmodels"
)

# 模型用resnet18
pretrained_net = models.resnet18()
# 为我的模型加载参数
pretrained_net.load_state_dict(weights)
# 打印全连接层
print(pretrained_net.fc)

# 将fc层进行更改，此时会对fc层随机初始化
pretrained_net.fc = nn.Linear(512, 2)
# 打印新的全连接层，此时已被随机初始化了，但其他层没变
print(pretrained_net.fc)

# ********************************************************************************
# 由于是在很大的ImageNet数据集上预训练的，所以其他层参数已经足够好
# 因此一般只需使用较小的学习率来微调这些参数
# fc中的随机初始化参数一般需要更大的学习率从头训练
# 将fc的学习率设为已经预训练过的部分的10倍
# ********************************************************************************

# 以下将参数分开为fc层参数和特征层参数
# 取出最后一层 fc 的所有参数对象，并用 id() 转成它们在 Python 中的对象身份标识
# 也就是在output_params中存储所有fc层参数的id
output_params = list(map(id, pretrained_net.fc.parameters()))
# 在整个模型 pretrained_net.parameters() 里过滤掉属于 fc 的参数，得到其他层的参数
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       # fc层的参数的学习率是默认的10倍
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       # 默认学习率参数0.01
                       lr=lr, weight_decay=0.001)

# ========================
# 微调模型
# ========================

def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    # 训练集迭代器
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs),
                            batch_size, shuffle=True)
    # 测试集迭代器
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs),
                           batch_size)
    # 交叉熵损失函数
    loss = torch.nn.CrossEntropyLoss()
    # 调用
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

train_fine_tuning(pretrained_net, optimizer)

# 作为对比，一个相同的模型但所有参数都初始化了
# 学习率调大一点
scratch_net = models.resnet18(pretrained=False, num_classes=2)
lr = 0.1
optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
train_fine_tuning(scratch_net, optimizer)

