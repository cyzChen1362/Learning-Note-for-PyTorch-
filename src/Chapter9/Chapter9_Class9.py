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
    语义分割和数据集
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html
# ********************************************************************************

# ========================
# Pascal VOC2012 语义分割数据集
# ========================

import os
import torch
import torchvision
import d2lzh_pytorch as d2l
import shutil


d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = r"D:\LearningDeepLearning\LearningNote_Dive-into-DL-PyTorch\data\VOC2012"

def read_voc_images(voc_dir = r"D:\LearningDeepLearning\LearningNote_Dive-into-DL-PyTorch\data\VOC2012"
                    ,is_train=True):
    """
    读取 PASCAL VOC 图像及分割标注
    1. 先检查本地固定目录是否已经存在数据
    2. 如果不存在，再调用 d2l.download_extract 下载并复制
    3. 最后按原逻辑读取图像和标注
    """
    target_dir = voc_dir

    # 如果目标目录不存在或内容为空，再去下载
    if not (os.path.exists(target_dir) and os.listdir(target_dir)):
        tmp_dir = d2l.download_extract('VOC2012')  # 只在需要时下载
        os.makedirs(target_dir, exist_ok=True)
        for item in os.listdir(tmp_dir):
            s = os.path.join(tmp_dir, item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    # 根据 is_train 读取 train.txt/val.txt
    txt_fname = os.path.join(target_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        names = f.read().split()

    features, labels = [], []
    for fname in names:
        img_path = os.path.join(target_dir, 'JPEGImages', f'{fname}.jpg')
        lbl_path = os.path.join(target_dir, 'SegmentationClass', f'{fname}.png')
        features.append(torchvision.io.read_image(img_path))
        labels.append(torchvision.io.read_image(lbl_path, mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)

n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n)
d2l.plt.show()

#@save
# VOC颜色映射表
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
# VOC类别名称列表
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

#@save
# RGB标注图 —— 每个像素的类别索引图
def voc_colormap2label():
    """
        构建从RGB数到VOC类别索引的映射
        用途:
        给定一张 VOC 标注图 label_img（形状 H×W×3），可以快速得到类别图：
        def voc_label_indices(label_img, colormap2label):
             # label_img: (H, W, 3) uint8
            idx = (label_img[:,:,0]*256 + label_img[:,:,1])*256 + label_img[:,:,2]
            return colormap2label[idx]
    """
    # 建一个长度为 256^3 = 16777216 的一维张量，初值全 0
    # 每个可能的 RGB 组合对应一个唯一的索引位置
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    # 遍历 21 种 VOC 颜色
    for i, colormap in enumerate(VOC_COLORMAP):
        # 将一个 RGB 颜色映射到 0~16,777,215 的整数
        # 公式相当于把 (R,G,B) 当作三位 256 进制数：
        # index = R * 256^2 + G * 256 + B
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """
        将VOC标签中的RGB值映射到它们的类别索引。
        如果在CPU上使用numpy，与原逻辑一致；
        如果在GPU上使用tensor计算，避免数据搬运。
        Args:
            colormap:一张标注图的张量，形状通常是 (3, H, W)
            colormap2label:长度 256³ 的一维查表向量
        Return:
            colormap2label[idx]: 形状 (H, W)，每个元素是 0~20 的类别编号（与 VOC_CLASSES 对应）

    """
    if colormap.device.type == 'cuda:0':
        # ---- GPU: 全部使用tensor运算 ----
        colormap = colormap.long().permute(1, 2, 0)  # (H, W, 3)
        idx = (colormap[..., 0] * 256 + colormap[..., 1]) * 256 + colormap[..., 2]
        colormap2label = colormap2label.to(colormap.device)
        return colormap2label[idx]
    else:
        # ---- CPU: 保持原来的numpy逻辑 ----
        # permute(1, 2, 0)：把张量从 (C, H, W) 调整为 (H, W, C)，方便按像素访问 R/G/B
        colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
        # idx 形状为 (H, W)
        idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
               + colormap[:, :, 2])
        return colormap2label[idx]

y = voc_label_indices(train_labels[0], voc_colormap2label())
print(y[105:115, 130:140], VOC_CLASSES[1])

# ========================
# 预处理数据
# ========================
#@save
def voc_rand_crop(feature, label, height, width):
    """
        随机裁剪特征(feature)和标签(label)图像
        作用是：在数据增强时，对输入图像 feature 和对应的标注 label 同步做同样的随机裁剪
        Args:
            feature:待处理的输入图像（通常是 RGB 图），形状 (C, H, W)
            label:对应的分割标注图（每像素类别索引或 RGB），形状与 feature 高宽一致
            height:希望裁剪得到的输出图像的高度
            width:希望裁剪得到的输出图像的宽度
        Return:
            feature: 随机位置裁剪出来的图像张量，形状为 (C, height, width)
            label: 随机位置裁剪出来的标签张量，形状为 (C, height, width)

    """
    # rect：(top, left, new_height, new_width)，表示从feature中随机选取的裁剪区域
    # 其中top是从原图的最上边开始往下的像素数，left是从左往右
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    # functional.crop：按照给定的 top, left, height, width 参数裁剪图像
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n)
d2l.plt.show()

# ========================
# 自定义语义分割数据集类
# ========================
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # crop_size：例如 (320, 480)，后续随机裁剪时的高宽
        self.crop_size = crop_size
        # features：列表，其中每个元素都是一张原始RGB图像(C×H×W)
        # labels：列表，其中每个元素都是对应的标注图（每个像素是类别颜色的 RGB 图）
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        # filter：过滤掉尺寸小于 crop_size 的图像，确保随机裁剪时不会越界。
        # normalize_image：先把像素缩放到[0,1]，再做标准化(x-mean)/std，
        # 使用的是 ImageNet 常见均值和方差
        # colormap2label：生成颜色到类别索引的一维查表张量，用于后续把标注RGB转成类别ID
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    # 将 0–255 的 uint8 张量转为 float32，再除以 255 归一化到 0–1
    # 再用 Normalize 做标准化
    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    # 只保留高 ≥ crop_size[0] 且宽 ≥ crop_size[1] 的图
    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    # 对第idx个图像随机裁剪voc_rand_crop，并对标签同步裁剪成 crop_size。
    # 标签转索引：voc_label_indices 把裁剪后的 RGB 标注图转成 (H, W) 的类别索引图
    # 每个元素都是这个像素对应的类别索引
    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

# ========================
# 读取数据集
# ========================

crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)

batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=0)
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break

# ========================
# 整合所有组件
# ========================
#@save
def load_data_voc(batch_size, crop_size,
                  voc_dir=r"D:\LearningDeepLearning\LearningNote_Dive-into-DL-PyTorch\data\VOC2012"):
    """
    加载VOC语义分割数据集
    1. 如果指定目录下已有数据且非空，则直接使用
    2. 否则下载并解压到默认位置
    """
    # 如果本地目录不存在或为空才下载
    if not (os.path.exists(voc_dir) and os.listdir(voc_dir)):
        print(f"[INFO] 未检测到数据集，开始下载...")
        voc_dir = d2l.download_extract('voc2012', os.path.join('VOCdevkit', 'VOC2012'))
    else:
        print(f"[INFO] 使用已有数据集：{voc_dir}")

    # DataLoader能传入VOCSegDataset类的原因：
    # DataLoader 需要的只是：
    # class MyDataset(torch.utils.data.Dataset):
    #     def __getitem__(self, idx):
    #         ... 给定一个索引 idx，返回单个样本 (feature, label)
    #     def __len__(self):
    #         ... 告诉 DataLoader 这个数据集有多少个样本

    num_workers = 0 # 因为我是高贵的windows所以不需要num_workers。

    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir),
        batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir),
        batch_size,
        drop_last=True,
        num_workers=num_workers
    )
    return train_iter, test_iter

