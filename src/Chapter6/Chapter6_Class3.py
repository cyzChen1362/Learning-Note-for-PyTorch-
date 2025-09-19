"""
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
    语言模型数据集（周杰伦专辑歌词）
"""

# ========================
# 读取数据集
# ========================

import torch
import random
import zipfile

with zipfile.ZipFile('../../data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
print(corpus_chars[:40])

# 把换行符替换成空格
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
# 仅使用前1万个字符来训练模型
corpus_chars = corpus_chars[0:10000]

# ========================
# 建立字符索引
# ========================

# ********************************************************************************
# 以下代码封装在d2lzh_pytorch包里的load_data_jay_lyrics函数中
# 调用该函数后会依次得到
# corpus_indices、char_to_idx、idx_to_char和vocab_size这4个变量
# ********************************************************************************

# 把字符串视为字符序列，提取其中不重复的字符
# 字符到索引的映射
# 这一步是随机的
idx_to_char = list(set(corpus_chars))
# 索引到字符的反向映射
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
print(vocab_size) # 1027

# 将训练数据集中每个字符转化为索引，并打印前20个字符及其对应的索引
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)

# ========================
# 时序数据的采样
# ========================

# ********************************************************************************
# 时序数据的一个样本通常包含连续的字符
# 假设时间步数为5，样本序列为5个字符，即“想”“要”“有”“直”“升”
# 该样本的标签序列为这些字符分别在训练集中的下一个字符，即“要”“有”“直”“升”“机”
# 有两种方式对时序数据进行采样，分别是随机采样和相邻采样
# ********************************************************************************

# ========================
# 随机采样
# ========================

# 从数据里随机采样一个小批量
# 其中批量大小batch_size指每个小批量的样本数，num_steps为每个样本所包含的时间步数
# 每个样本是原始序列上任意截取的一段序列。相邻的两个随机小批量在原始序列上的位置不一定相毗邻
# 因此，我们无法用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态
# 在训练模型时，每次随机采样前都需要重新初始化隐藏状态

"""
    example:
    "周杰伦的音乐很好听……"
    num_steps = 5
    [周 杰 伦 的 音]
    [杰 伦 的 音 乐]
    ...
    如果batch_size = 3：一次取3条这样的序列拼成一个批次，形成张量形状 [3, 5]
    
    num_steps 控制RNN 展开的时间步（也就是链子的长度）
    batch_size 控制一次处理多少条序列（一次性算多少条链子）
    epoch_size 这一批并行链子训练多少次才训练完整个语料库的一个epoch
    
"""

"""
    可以理解为：
    data_iter_random是每次输出：
    batch_size * num_steps 的数据，一共输出epoch_size次
    每一行内的时序是连续的，但不同行以及epoch_size都是割裂的

"""

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为输出的索引x是相应输入的索引y加1
    # num_examples是总输出样本数
    # 输出在输入之后的一个开始输出，所以当然是总输出样本数比总样本数少一个
    num_examples = (len(corpus_indices) - 1) // num_steps
    # it is clear that
    epoch_size = num_examples // batch_size
    # 每一条的索引并打乱
    # 这里存放的只是“第几个样本”，而不是样本在原始序列里的字符下标
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        # corpus_indices存储的是顺序字符的索引
        return corpus_indices[pos: pos + num_steps]

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        # 也就是读多少条做并行
        i = i * batch_size
        # example_indices 这里存放的只是“第几个样本”，而不是样本在原始序列里的字符下标
        # 所以j自然要乘以num_steps，这样才能对应上_data里的原始字符索引列表corpus_indices
        # 按条来抽取
        # batch_indices告诉你要从第几条抽到第几条
        batch_indices = example_indices[i: i + batch_size]
        # 遍历抽每条，for写到lambda里面了
        # X是输入数据条，Y是输出数据条
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)

# 输入一个从0到29的连续整数的人工序列
# 可见，相邻的两个随机小批量在原始序列上的位置不一定相毗邻
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=3, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')

# ========================
# 相邻采样
# ========================

# 很显然，就是令相邻的两个随机小批量在原始序列上的位置相毗邻
# 这时候，我们就可以用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态
# 从而使下一个小批量的输出也取决于当前小批量的输入，并如此循环下去

# ********************************************************************************
# 前向传播：
# 在相邻采样下，第 k 个小批量的初始隐藏状态直接沿用第 k-1 个小批量的最后隐藏状态
#
# 反向传播：
# 如果不做任何处理，计算图会从第1个批量一路串到最后
# 反向传播时梯度会从最后一个批量一直回传到最前面，图会越来越大，内存和计算量失控
# 实际训练会“截断反向传播”(Truncated BPTT)：
# 反向传播只在当前批量（有时加上一个可控长度的窗口）内回传，不会无休止地追溯到第一个批量
# ********************************************************************************

"""
    可以理解为：
    data_iter_consecutive是：
    先整出batch_size * batch_len的一个大矩阵
    然后将这个矩阵每num_steps列割成一个小矩阵进行输出
    一共输出epoch_size = (batch_len - 1) // num_steps个小矩阵
    其中第epoch_size{i}的第k行的最后一个，和epoch_size{i+1}的第k行的第一个是时序相连的
    所以这个RNN的这一条可以顺序训练，继承隐藏状态

"""

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # corpus_indices存储的是顺序字符的索引
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    # 把总长度 data_len 均分成 batch_size 条连续的长序列
    # 每条长序列的长度是 batch_len
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    # 截取能整除的部分，然后reshape成[batch_size, batch_len]
    # 每一行都是一条连续的长序列
    # 相邻两行的末尾和开头虽然在原始序列上是相邻的字符，但在这个实现里不会跨行传递隐藏状态
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    # 只是在每一行内部按列切片
    # RNN 前向时通常会把每一行当作一个独立样本
    # 在这行内部的时间步是连续的，但行与行之间完全不共享隐藏状态
    # 也就是并行计算，效率提升但跨行依赖减少
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

# 打印
for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')


