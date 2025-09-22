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


d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

# ========================
# 读取数据集
# ========================

def read_data_bananas(is_train=True):
    # 先按照 d2l 默认规则下载
    tmp_dir = d2l.download_extract('banana-detection')

    # 希望的最终路径
    target_dir = r"D:\LearningDeepLearning\LearningNote_Dive-into-DL-PyTorch\data\bananas"
    os.makedirs(target_dir, exist_ok=True)

    # 如果目标目录为空，则移动一次即可
    if not os.listdir(target_dir):
        for item in os.listdir(tmp_dir):
            s = os.path.join(tmp_dir, item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    sub_dir = 'bananas_train' if is_train else 'bananas_val'
    csv_fname = os.path.join(target_dir, sub_dir, 'label.csv')

    csv_data = pd.read_csv(csv_fname).set_index('img_name')

    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        img_path = os.path.join(target_dir, sub_dir, 'images', img_name)
        images.append(torchvision.io.read_image(img_path))
        targets.append(list(target))

    return images, torch.tensor(targets).unsqueeze(1) / 256

# 本函数已保存在d2lzh_pytorch包中方便以后使用
class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

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
