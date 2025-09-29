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
    风格迁移
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_computer-vision/neural-style.html
# 解释一下那张图：
# 那张图的三列卷积层是完全相同，并且参数已经预训练完毕，并且不更改了的
# 首先传入内容图像和样式图像，然后三个卷积层分别抽取特征
# 然后随机初始化一个合成图像，同样经过三个卷积层抽取特征
# 然后如图，合成图像列第一、三层的卷积层特征和样式图像对应层特征做损失
# 合成图像列第二层的卷积层特征和内容图像对应层特征做损失
# 再加上合成图像的总变差损失（其实就是每个像素和右、下的像素的像素值差）
# 三个损失一起训练，注意训练的不是任何一个模型的参数，而是调整合成图像x
# 就类似于：正常 y_hat = w*x + b，loss = y - y_hat，是对w和b求梯度并训练
# 而这里是对x求梯度并训练，其他不变
# ********************************************************************************

# ========================
# 阅读内容和风格图像
# ========================

import torch
import torchvision
from torch import nn
import d2lzh_pytorch as d2l

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

d2l.set_figsize()
# content_img = d2l.Image.open('../../data/img/rainier.jpeg')
content_img = d2l.Image.open('../../data/img/Building_Sea.jpg')
d2l.plt.imshow(content_img);
d2l.plt.show()

# style_img = d2l.Image.open('../../data/img/autumn-oak.jpeg')
style_img = d2l.Image.open('../../data/img/Plain_near_Auvers.jpg')
d2l.plt.imshow(style_img);
d2l.plt.show()

# ========================
# 预处理和后处理
# ========================

# ImageNet 数据集上常用的三个通道（R、G、B）像素均值与标准差
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

# 前处理函数，把一张普通的图像转成神经网络可以接受的张量
def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        # 调整尺寸
        torchvision.transforms.Resize(image_shape),
        # 换成张量，且HWC--CHW，且0-255-->0-1
        torchvision.transforms.ToTensor(),
        # 标准化
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    # 增加 batch 维度
    return transforms(img).unsqueeze(0)

def postprocess(img):
    # 去掉 batch 维并转device
    img = img[0].to(rgb_std.device)
    # CHW--HWC并逆标准化，调整通道只是因为(3,) 自动视为 (1, 1, 3)，方便计算
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    # 又重新调整回CHW并交给PIL.Image 对象
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

# ========================
# 抽取图像特征
# ========================

# pretrained_net = torchvision.models.vgg19(pretrained=True)

# 预模型参数下载到指定地址 ..\..\data\pretrainedmodels
# 同时显式写出url

from torch.hub import load_state_dict_from_url
# 预模型参数
weights = load_state_dict_from_url(
    'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    model_dir=r"..\..\data\pretrainedmodels"
)

# 模型用vgg19
pretrained_net = torchvision.models.vgg19()
# 为我的模型加载参数
pretrained_net.load_state_dict(weights)
# vgg中卷积层的索引可以通过打印pretrained_net实例获取
print(pretrained_net)

# 越靠近输入层，越容易抽取图像的细节信息；反之，则越容易抽取图像的全局信息
# 选择第四卷积块的最后一个卷积层作为内容层
# 选择每个卷积块的第一个卷积层作为风格层
# 当然也可以自己换图片然后改改content_layers
# 算上ReLu MaxPooling这些肯定不止19
style_layers, content_layers = [0, 5, 10, 19, 28], [19]

# 找到最高层（28）并列表推导式将0-28层依次取出，这些曾作为参数传入nn.Sequential
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])

def extract_features(X, content_layers, style_layers):
    """
        逐层计算并保留内容层和风格层的输出
        Args:
            X:输入张量，形状 [1, 3, H, W]，经过 preprocess 处理后的图像
            content_layers:列表，如 [19]，指定在哪些层提取“内容”特征
            style_layers:列表，如 [0, 5, 10, 19, 28]，指定在哪些层提取“风格”特征
        Return:
            contents:用来存放内容特征图
            styles:用来存放风格特征图
        """
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

# 函数如其名，不必解释
def get_contents(image_shape, device):
    """
        内容图像特征函数
        Args:
            image_shape:图片形状
            device:设备
        Return:
            content_X:内容图像
            contents_Y:内容图像的内容特征
        """
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

# 函数如其名，不必解释
def get_styles(image_shape, device):
    """
        样式图像特征函数
        Args:
            image_shape:图片形状
            device:设备
        Return:
            style_X:样式图像
            styles_Y:样式图像的样式特征
        """
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y

# ========================
# 定义损失函数
# ========================

# 内容损失
def content_loss(Y_hat, Y):
    """
        内容损失函数
        Args:
            Y_hat:合成图像内容特征
            Y:内容图像内容特征
        Return:
            content_loss:合成图像内容特征损失
        """
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量。
    return torch.square(Y_hat - Y.detach()).mean()

# 风格损失，使用Gram矩阵，数学原理看书，很简单的
def gram(X):
    """
        Gram矩阵计算函数
        Args:
            X:输入原始图像
        Return:
            gram_X:输入Gram矩阵图像
        """
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    """
        风格损失计算函数
        Args:
            Y_hat:合成图像样式特征
            gram_Y:样式图像样式特征gram矩阵
        Return:
            style_loss:合成图像内容特征损失
        """
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

# 全变分损失
def tv_loss(Y_hat):
    """
        全变分损失计算函数
        Args:
            Y_hat:合成图像
        Return:
            tv_loss:合成图像全变分损失
        """
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

# 损失函数权重
content_weight, style_weight, tv_weight = 1, 1e4, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    """
        总损失计算函数
        Args:
            X:合成图像
            contents_Y_hat:合成图像的内容特征
            styles_Y_hat:合成图像的样式特征
            contents_Y:内容图像的内容特征
            styles_Y_gram:样式图像的样式特征
        Return:
            compute_loss:总损失计算
        """
    # 分别计算内容损失、风格损失和全变分损失
    # 这里用zip是因为contents_Y_hat, contents_Y/styles_Y_hat,
    # styles_Y_gram可能记录的是多个层输出的特征，所以用zip来逐层获取
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l

# ========================
# 初始化合成图像
# ========================
class SynthesizedImage(nn.Module):
    # 继承自 nn.Module，但没有任何卷积层或线性层，只有一个可学习参数 self.weight
    # img_shape 形如 (1, 3, H, W)
    # nn.Parameter 使得 weight 会自动出现在 model.parameters() 中，从而能被优化器更新
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))
    # forward 直接返回 self.weight，即：前向传播的输出就是这张图像本身
    def forward(self):
        return self.weight

def get_inits(X, device, lr, styles_Y):
    """
        合成图像初始化函数
        Args:
            X:放入的初始化图像，这里选择放入内容图像
            device:设备
            lr:学习率
            styles_Y:样式图像的样式特征
        Return:
            gen_img():初始化的图像，其为自定义SynthesizedImage类
            styles_Y_gram：样式图像的的样式特征的Gram矩阵
            trainer:优化器
        """
    # 创建可训练图像
    gen_img = SynthesizedImage(X.shape).to(device)
    # 初始化为内容图像
    gen_img.weight.data.copy_(X.data)
    # Adam 优化器会更新 gen_img.weight
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    # 你看，y_hat是gram，y也是gram
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer

# ========================
# 训练模型
# ========================

import matplotlib.pyplot as plt

def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    """
        训练函数
        Args:
            X:放入的初始化图像，这里选择放入内容图像
            contents_Y:内容图像的内容特征
            styles_Y:样式图像的样式特征
            device:设备
            lr:学习率
            num_epochs:epoch数
            lr_decay_epoch:每隔几个epoch衰减一次学习率
        Return:
            X:训练结束后的合成图像
        """
    # 初始化
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)

    # trainer 配置一个分段学习率调度器，让学习率在训练过程中按照固定步长衰减
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)

    # 用列表存储 loss
    epochs, content_losses, style_losses, tv_losses = [], [], [], []

    for epoch in range(num_epochs):
        trainer.zero_grad()
        # 合成图像的内容特征和样式特征
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        # 合成图像的内容特征损失、样式特征损失、全变分损失、总损失
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        # 更新参数
        l.backward()
        trainer.step()
        scheduler.step()

        # 每 10 轮记录一次
        if (epoch + 1) % 10 == 0:
            epochs.append(epoch + 1)
            content_losses.append(float(sum(contents_l)))
            style_losses.append(float(sum(styles_l)))
            tv_losses.append(float(tv_l))

    # 训练结束后一次性画图
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, content_losses, label='content')
    plt.plot(epochs, style_losses, label='style')
    plt.plot(epochs, tv_losses, label='TV')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Style Transfer Loss Curves')
    plt.show()

    # 也可以在此处显示最终合成的图像
    plt.figure(figsize=(6, 4))
    plt.imshow(postprocess(X))
    plt.axis('on')
    plt.show()

    return X

image_shape = (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
