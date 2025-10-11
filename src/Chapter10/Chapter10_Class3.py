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
    序列到序列学习（seq2seq）
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_recurrent-modern/seq2seq.html
# b站：https://www.bilibili.com/video/BV16g411L7FG?spm_id_from=333.788.recommend_more_video.0&vd_source=8086d55c4f00a2130d3e31cecb0db076
# ********************************************************************************

import collections
import math
import torch
from torch import nn
import d2lzh_pytorch as d2l

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ========================
# 编码器
# ========================

#@save
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    # Seq2SeqEncoder = embedding + rnn
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # 其中每个batch是一串字符的每个字符的隐藏状态
        # state的形状:(num_layers,batch_size,num_hiddens)
        # 其中每个batch是最后一个字符的每一层的隐藏状态
        return output, state

# 实例化上述编码器的实现
# encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
#                          num_layers=2)
# encoder.eval()
# X = torch.zeros((4, 7), dtype=torch.long)
# output, state = encoder(X)
# print(output.shape)
# print(state.shape)

# ========================
# 解码器
# ========================

class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    # Seq2SeqDecoder = ( embedding concat(embed_size维) state[-1] ) + rnn
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # 这里进行修改,将state初始化变为一个tuple
        return (enc_outputs[1], enc_outputs[1][-1])

    # 这里进行修改：
    # 源代码中predict_seq2seq的
    # Y, dec_state = net.decoder(dec_X, dec_state)
    # 说明下一个字的X将会concat上一个字的state的最顶层隐状态
    # 然后self.rnn(X_and_context, state)的state同时又是上一个字的state
    # 但其实我们(Bahadanau)希望的是，X去concat源句子的最终输出的state的最顶层隐状态
    # 然后self.rnn(X_and_context, state)的state是上一个字的state
    # 所以进行如下修改（参考别人的）
    def forward(self, X, state):
        # state的形状：([num_layers, batch_size, num_hiddens],[batch_size,num_hiddens])
        # state[-1]即最后一层
        # embedding输出'X'的形状：(batch_size,num_steps,embed_size),换维
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        # new
        encode = state[1] # 2D
        state = state[0]  # 3D
        # new end
        X_and_context = torch.cat((X, context), 2)

        # self.rnn的输入：
        # X_and_context → (num_steps, batch_size, embed_size + num_hiddens)
        # state → (num_layers, batch_size, num_hiddens)
        # self.rnn的输出：
        # output：形状 (num_steps, batch_size, num_hiddens)
        # state：形状 (num_layers, batch_size, num_hiddens)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, (state, encode)

# 实例化上述解码器的实现
# decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
#                          num_layers=2)
# decoder.eval()
# state = decoder.init_state(encoder(X))
# output, state = decoder(X, state)
# print(output.shape)
# print(state.shape)

# ========================
# 损失函数
# ========================

#@save
# 在序列中屏蔽不相关的项
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    # 将超过“有效长度”的部分（即 <pad> 位置）用指定值 value 替换
    # 取出句子长度（第二个维度），即每个句子的最大时间步 num_steps
    maxlen = X.size(1)

    # 假设: valid_len, maxlen = tensor([3, 5]), 7
    # 则: torch.arange(maxlen)  →  tensor([0,1,2,3,4,5,6])
    # mask =
    # [[0,1,2,3,4,5,6] < 3] → [True, True, True, False, False, False, False]
    # [[0,1,2,3,4,5,6] < 5] → [True, True, True, True, True, False, False]
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # ~mask 表示逻辑取反，即无效位置
    X[~mask] = value
    return X

# 测试
# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(sequence_mask(X, torch.tensor([1, 2])))

#@save
# 带遮蔽的softmax交叉熵损失函数
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        # 创建遮蔽权重矩阵
        # label.shape = [2, 6]
        # valid_len = [3, 5]
        # weights =
        # [[1, 1, 1, 0, 0, 0],
        #  [1, 1, 1, 1, 1, 0]]
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        # 关闭 PyTorch 内部的自动平均, 防止填充的部分影响整体平均损失
        self.reduction='none'
        # PyTorch 的 CrossEntropyLoss 期望输入形状为 (batch_size, vocab_size, num_steps)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

# 测试
# loss = MaskedSoftmaxCELoss()
# print(loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
#      torch.tensor([4, 2, 0])))

# ========================
# 训练
# ========================

import time
import matplotlib.pyplot as plt

#@save
# Seq2Seq训练
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    # 对模型中的 nn.Linear 和 nn.GRU 层的 weight 参数使用 Xavier 均匀初始化
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    # 权重初始化
    net.apply(xavier_init_weights)

    # device，优化，损失，开启训练
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    # 用列表记录每个 epoch 的平均 loss
    epoch_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()  # 使用 time 包计时
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量

        for batch in data_iter:
            optimizer.zero_grad()
            # 见 10.1 load_data_nmt
            # X, Y: (batch_size, num_steps)
            # X_valid_len, Y_valid_len: (batch_size,)
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 对每个样本加上句首符号 <bos>
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            # 并去掉原句的最后一个词
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            # 这里dec_input是传给net的第二个参数
            # 也就是EncoderDecoder的forward的dec_X参数
            # 也就是Seq2SeqDecoder的forward的X参数
            # 也就是强制教学
            # ********************************************************************************
            # 这里一个很细的问题是：self.rnn会不会遇到不同的num_step的x输入的情况？
            # 以及如果是不同num_step的x，是自适应停下吗？
            # 实质上并不会，因为num_step用了truncate_pad方法强制对齐等长
            # rnn对于所有等长的输入x都是机械地走完num_step而已
            # 同时MaskedSoftmaxCELoss遮蔽机制忽略了计算损失时填充的部分
            # 同时，也不需要显式告诉self.rnn的num_step有多大
            # X = self.embedding(X)
            # X = X.permute(1, 0, 2)
            # output, state = self.rnn(X)
            # 这里rnn会按输入张量的第一维的长度展开时间步
            # ********************************************************************************
            Y_hat, _ = net(X, dec_input, X_valid_len)
            # 损失函数、反向传播
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            # 裁剪梯度，防止梯度爆炸
            d2l.grad_clipping(net.parameters(), 1, device)
            # 当前 batch 的有效词元总数为 num_tokens
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)

        avg_loss = metric[0] / metric[1]
        epoch_losses.append(avg_loss)
        end_time = time.time()
        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}, loss {avg_loss:.3f}')

    # 绘制 loss 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_epochs + 1), epoch_losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()

    # 结束后打印最后一次 loss 与速度
    total_tokens = metric[1]
    total_time = end_time - start_time
    print(f'loss {avg_loss:.3f}, {total_tokens / total_time:.1f} tokens/sec on {str(device)}')

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs = 0.005, 300

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
# len(src_vocab)即vocab_size
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

# ========================
# 预测
# ========================

#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    # src_sentence是源句子，将源句子变成小写并分词，然后src_vocab是源句子的词表
    # 也就是源句子分割为词并转为索引形式
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    # 源vocab_size
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    # 填充或截断为num_step
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴batch_size
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    # encoder
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    # 这句将encoder的output,state传入了decoder的init_state中
    # net.decoder.init_state返回元组(enc_outputs[1],enc_outputs[1][-1])
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴，初始化第一个输出X
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        # 这里其实是要改的，但直接改上面的Seq2SeqDecoder类了
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        # 训练阶段self.rnn的时间步是num_step
        # 但预测阶段不需要，预测阶段当output是<eos>的时候直接break即可
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

# ========================
# 预测序列的评估
# ========================

def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    # 对着算式看吧，k是其中一个评估参数，根据实际情况选择
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        # collections.defaultdict(int)：
        # 和普通的 dict 类似，但在访问不存在的键时不会报错，而是自动创建一个默认值
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            # 如果某个 n-gram 第一次出现，defaultdict 自动给它赋值 0；然后再执行 += 1，变成 1
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
