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

import torch
from torch import nn

"""
    GPU计算
"""

# =========================
# 1. 计算设备
# =========================

"""
# torch版本
print(torch.__version__)
# 查看cuda+GPU是否可用
print(torch.version.cuda)
print(torch.cuda.is_available())
# 查看GPU数量
print(torch.cuda.device_count())
# 查看当前GPU索引号
print(torch.cuda.current_device())
# 根据索引号查看GPU名字
print(torch.cuda.get_device_name(0))
# 支持算力列表
print(torch.cuda.get_arch_list())

"""

"""
# 以下操作在终端实现：

# 查看NVIDIA显卡信息
nvidia-smi

# Anaconda新建一个干净的环境：
# 当然这个名字可以改一下
conda create -n torch121 python=3.9
conda activate torch121

# 在 PyCharm 里让终端自动用 Anaconda Prompt
# 步骤一：找到你的 Anaconda Prompt 路径
E:/anaconda3/Scripts/activate.bat
# 步骤二：设置 PyCharm 的终端默认 shell
1. PyCharm -> Settings（设置）
2. 搜索 terminal
3. 找到 Shell path 或 终端 -> Shell 路径
4. 设定为上面那个地址

# 解释器也要用对
1. 右下角设置 Python 解释器为 E:/anaconda3/envs/torch121/python.exe （当然不同环境的环境名和路径也不一样）
2. 确认python 路径：where python

# 这里是安装CUDA 12.1 版 PyTorch：
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# if 当前 PyTorch 版本（预编译版） 不支持 你的显卡的计算能力（sm_120），这样做：
# 安装torch 2.7.0 + cu128
# 可以直接在当前conda 环境的终端操作
# 卸载老版 torch 相关包（保险起见）
pip uninstall torch torchvision torchaudio
# torch 2.7.0 + cu128有，但torchvision 0.18.0 + cu128没有对应的版本：
pip install torch==2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 可能会发现torchvision和torch版本不匹配的问题
# 这个网址可以看兼容表
https://pytorch.org/get-started/previous-versions/
# 执行这个指令即可
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 重命名conda环境，例如从“torch121”换成"torch270cu128"
# 克隆你的环境到新名字
conda create --name torch270cu128 --clone torch121
# 确认新环境可用
conda activate torch270cu128
# 右下角切换解释器
...
# 运行下代码
...
# 测试没问题后，删除原来的环境
conda remove --name torch121 --all

# OSError: [WinError 127] 找不到指定的程序
# 典中典之torchtext版本过低和当前高版本torch不匹配
# 然而torchtext已经一年没更新了，不太可能降低torch版本来适配torchtext
# 尤其是高算力架构sm120是需要高版本torch的
# 所以最好的方法就是不用torchtext
# 在d2lzh_pytorch/utils.py中将import torchtext和import torchtext.vocab as Vocab注释就好
# 后面别的地方再改改得了

# 关于OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.错误解决方法
https://zhuanlan.zhihu.com/p/371649016

"""

# =========================
# 2. Tensor的GPU计算
# =========================
# 默认情况下，Tensor会被存在内存上
x = torch.tensor([1, 2, 3])
print(x)
# 使用.cuda()可以将CPU上的Tensor转换（复制）到GPU上
x = x.cuda(0)
print(x)
# 通过Tensor的device属性来查看该Tensor所在的设备
print(x.device)
# 直接在创建的时候就指定设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1, 2, 3], device=device)
print(x)
# or
x = torch.tensor([1, 2, 3]).to(device)
print(x)
# 如果对在GPU上的数据进行运算，那么结果还是存放在GPU上
y = x**2
print(y)
# 存储在不同位置中的数据是不可以直接进行计算的
# 即存放在CPU上的数据不可以直接与存放在GPU上的数据进行运算
# 位于不同GPU上的数据也是不能直接进行计算
# z = y + x.cpu()
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

# =========================
# 3. 模型的GPU计算
# =========================
# 模型默认在cpu上
net = nn.Linear(3, 1)
print(list(net.parameters())[0].device)
# 将其转换到GPU上
net.cuda()
print(list(net.parameters())[0].device)
# 同样的，需要保证模型输入的Tensor和模型都在同一设备上，否则会报错
x = torch.rand(2,3).cuda()
print(net(x))



