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
    机器翻译与数据集
"""

import os
import shutil
import collections
import math
import torch
from torch import nn
import d2lzh_pytorch as d2l

# ========================
# 下载和预处理数据集
# ========================

#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """载入“英语－法语”数据集"""
    tmp_dir = d2l.download_extract('fra-eng')

    # 希望的最终路径，在这个路径下创建文件夹
    target_dir = r"D:\LearningDeepLearning\LearningNote_Dive-into-DL-PyTorch\data\fra-eng"
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

    with open(os.path.join(target_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

# 测试
raw_text = read_data_nmt()
print(raw_text[:75])

#@save
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""

    def no_space(char, prev_char):
        # 如果当前字符 char 是标点符号之一（,、.、!、?），且前一个字符不是空格，则返回 True。
        # 用来判断是否需要在标点前插入一个空格
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格，并使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    # 在单词和标点符号之间插入空格
    # 如果当前字符是标点且前面没有空格 → 在标点前加一个空格；
    # 否则保持原样；
    # 最后把字符列表重新拼成字符串
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

# 测试
text = preprocess_nmt(raw_text)
print(text[:80])

# ========================
# 词元化
# ========================

#@save
# 把整段英法平行文本分割成 “词元（token）序列列表”
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    # source：英语句子词元化结果
    # target：法语句子词元化结果
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        # 用来限制只读取前 N 行，方便调试或快速预览
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            # 这里的split(' ')把句子词元化
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

# 测试
source, target = tokenize_nmt(text)
print(source[:6])
print(target[:6])

#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize(figsize=(8, 6))
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
    d2l.plt.show()

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);

# ========================
# 词表
# ========================

src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
print(len(src_vocab))

# ========================
# 加载数据集
# ========================

#@save
# 截断或填充文本序列
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    # 把输入的 token 序列 line 处理成固定长度 num_steps；
    # 太长就截断；
    # 太短就用 padding_token 填充。
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

#@save
# 句子 —— “张量 + 有效长度”
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    # 输入：
    # 若干句子（词序列）
    # 输出：
    # array：形状一致的 Tensor（张量）
    # valid_len：每个句子实际的有效长度（不包括 <pad>）

    # 利用 Vocab.__getitem__()，将每个句子的 token 列表转成整数索引；
    # [["i","love","you"],["you","love","me"]]
    # → [[3,5,4],[4,5,6]]
    lines = [vocab[l] for l in lines]

    # 给每个句子加上结束符 <eos>
    # [3,5,4] → [3,5,4,7]  # 若 <eos> 的索引为7
    lines = [l + [vocab['<eos>']] for l in lines]

    # 截断或填充
    # 使用 truncate_pad() 让所有句子等长；
    # 不足的部分用 <pad> 补齐；
    # 最后转为 PyTorch 张量。
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])

    # 计算每个句子的有效长度
    # (array != vocab['<pad>']) 生成布尔矩阵，True 表示非填充位置；
    # .sum(1) 沿着句子维度求和；
    # 结果是每个句子的实际词元数
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)

    return array, valid_len

# ========================
# 训练模型
# ========================

#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    # batch_size	每次迭代的样本数量
    # num_steps	    每个句子的固定长度（截断或填充）
    # num_examples	只使用前多少个样本（加快调试；默认 600）

    # 读取并预处理原始数据
    # 调用 read_data_nmt() 载入 “英-法” 原始文本，然后用 preprocess_nmt() 清洗
    text = preprocess_nmt(read_data_nmt())

    # 使用制表符 \t 拆分出英文 (source) 和法文 (target)，再按空格分词
    source, target = tokenize_nmt(text, num_examples)

    # 得到源语料的词表
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 得到目标语料的词表
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])

    # 将源语料的每个句子转成有效长度的张量，以及每句的有效长度
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    # 将目标语料的每个句子转成有效长度的张量，以及每句的有效长度
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)

    # 拼接四个张量
    # 这里data_arrays[1]和data_arrays[3]就是每一句的有效长度，迭代器也会给出
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    # 封装成批量数据迭代器
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break
