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
    目标检测和边界框
"""

# ********************************************************************************
# 很多时候图像里有多个我们感兴趣的目标
# 我们不仅想知道它们的类别，还想得到它们在图像中的具体位置
#
# 在目标检测里，我们通常使用边界框（bounding box）来描述目标位置
# 边界框是一个矩形框，可以由矩形左上角的x和y轴坐标与右下角的x和y轴坐标确定
# 图中的坐标原点在图像的左上角
# ********************************************************************************

# ========================
# 导入实验所需的包
# ========================

from PIL import Image

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

d2l.set_figsize()
img = Image.open('../../data/img/catdog.jpg')
d2l.plt.imshow(img);  # 加分号只显示图
d2l.plt.show()

# ========================
# 边界框
# ========================

# bbox是bounding box的缩写
dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]

def bbox_to_rect(bbox, color):  # 本函数已保存在d2lzh_pytorch中方便以后使用
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
d2l.plt.show()
