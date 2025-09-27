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
    目标检测数据集（香蕉）
"""

# ********************************************************************************
# 由于新书后面的部分章节不完善，这里直接使用原版书
# 新书使用的是皮卡丘数据集，原书使用的是香蕉数据集
# 所以这里跟随原书同样使用香蕉数据集
# 同时在d2l库里面也做一些改动
# 最终的数据集下载在data/bananas下
# ********************************************************************************

# ========================
# 下载数据集
# ========================

import os
import pandas as pd
import torch
import torchvision
import d2lzh_pytorch as d2l
import shutil
import numpy as np

d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

# ========================
# 读取数据集
# ========================

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def read_data_bananas(is_train=True):
    # 先按照 d2l 默认规则下载
    tmp_dir = d2l.download_extract('banana-detection')

    # 希望的最终路径，在这个路径下创建文件夹
    target_dir = r"D:\LearningDeepLearning\LearningNote_Dive-into-DL-PyTorch\data\bananas"
    os.makedirs(target_dir, exist_ok=True)

    # 如果目标目录为空，则移动一次即可
    # 将文件从默认规则下载的位置移动到目标路径
    if not os.listdir(target_dir):
        for item in os.listdir(tmp_dir):
            s = os.path.join(tmp_dir, item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    # 下面的逻辑和源代码完全相同

    # 根据 is_train 决定读 bananas_train 还是 bananas_val
    sub_dir = 'bananas_train' if is_train else 'bananas_val'
    # label.csv 是标注文件，存储每张图片的边界框信息
    csv_fname = os.path.join(target_dir, sub_dir, 'label.csv')

    # 用 pandas 读入 csv，img_name 列设为索引
    # 每行形如：
    # img_name,x_min,y_min,x_max,y_max
    # 0001.png,48,240,195,371
    # ...
    # 用了set_index('img_name')，也就是label.csv的index就是img_name列
    csv_data = pd.read_csv(csv_fname).set_index('img_name')

    # for idx, row in csv_data.iterrows():
    # 每次迭代返回 (index, Series) 这样的二元组：
    # idx：该行的索引值（这里就是 img_name，因为前面用 set_index('img_name')）。
    # row：该行数据，类型是 pandas.Series，键为列名，值为该行对应的数值。
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        # img_path：拼出图片完整路径
        img_path = os.path.join(target_dir, sub_dir, 'images', img_name)
        # 读取一张图片并追加到 images 列表，返回形状[C, H, W]
        images.append(torchvision.io.read_image(img_path))
        targets.append(list(target))

    # targets转成torch.Tensor，形状为[N, 1, 5] [batch, 1,  label xmin ymin xmax ymax]
    # 将坐标除以 256，实现归一化到 0–1 区间(数据集中图像尺寸是 256×256)
    return images, torch.tensor(targets).unsqueeze(1) / 256

# 本函数已保存在d2lzh_pytorch包中方便以后使用
class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    # DataLoader 取数据时会自动调用
    # 传入单个索引 idx，返回图像[3, H, W], 标签[1, 5]
    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    # 返回数据集样本数 N
    def __len__(self):
        return len(self.features)

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape)
print(batch[1].shape)

# ========================
# 演示
# ========================

# 将 [batch, 3, H, W] → [batch, H, W, 3] 并归一化
imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255

# 新版 show_images 返回二维 numpy.ndarray
axes = d2l.show_images(imgs, 2, 5, scale=2)

# 展平成一维，确保每个元素都是 matplotlib.axes._axes.Axes
axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

# 画出边界框：忽略类别，只取坐标并乘回像素大小
for ax, label in zip(axes, batch[1][:10]):
    bbox = label[0, 1:] * edge_size          # [xmin, ymin, xmax, ymax]
    rect = d2l.bbox_to_rect(bbox, color='w') # 或 color='r' 更清晰
    ax.add_patch(rect)
d2l.plt.show()
