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
    编码器-解码器架构
"""

# ********************************************************************************
# 原书：https://zh.d2l.ai/chapter_recurrent-modern/encoder-decoder.html
# b站：https://www.bilibili.com/video/BV1c54y1E7YP?spm_id_from=333.788.videopod.episodes&vd_source=8086d55c4f00a2130d3e31cecb0db076
# ********************************************************************************

# ========================
# 编码器
# ========================

from torch import nn

#@save
# Encoder 接口类
# 用途：为所有具体的编码器（RNN 编码器、CNN 编码器、Transformer 编码器等）提供一个统一的结构模板
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    # 调用父类的构造函数
    # **kwargs 允许子类传入任意额外参数（例如隐藏层大小、embedding维度、层数等）
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
    # forward() 是 必须由子类实现的抽象方法，定义了编码器的前向传播过程。
    # X：输入数据（比如一句话的词向量序列或图像特征）
    # *args：可选的额外参数（如有效长度mask等）
    # 返回编码后的结果（例如 RNN 的隐藏状态序列、Transformer 的最后层输出等）
    # 如果不在子类中实现 forward()，则会报错：
    def forward(self, X, *args):
        raise NotImplementedError

"""
    举例：RNN 编码器的子类实现:
    
    class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # X的形状：[batch_size, seq_len]
        X = self.embedding(X)               # -> [batch_size, seq_len, embed_size]
        X = X.permute(1, 0, 2)              # RNN输入要求(seq_len, batch_size, embed_size)
        output, state = self.rnn(X)
        # output [seq_len, batch, hidden_size]
        # state [num_layers, batch, hidden_size]
        return output, state
        
    # embedding = nn.Embedding(vocab_size=10000, embed_size=300)表示：
    # 词表大小（vocab_size）= 10000
    # → 一共 10000 个不同的词（每个词有一个唯一的整数 ID：0~9999）
    # 嵌入维度（embed_size）= 300
    # → 每个词被表示成一个 300 维的向量。
    
    # self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)：
    # https://chatgpt.com/s/t_68e78f0243208191ba876fe076e1a389
    # https://chatgpt.com/s/t_68e78f22170c8191a50cee7feb990269
    # https://chatgpt.com/s/t_68e78f36e83c819193a7be2bb4d9bd82
    # https://chatgpt.com/s/t_68e78f46775c81918b542909c7359e1e

"""

# ========================
# 解码器
# ========================

#@save
# Decoder 接口类
# 用途：为后续各种具体解码器（如RNN解码器、Transformer解码器）提供模板
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    # 调用父类的构造函数
    # **kwargs 允许子类传入任意关键字参数（如隐藏层维度、dropout等）
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    # 定义了解码器在开始工作前如何初始化内部状态（state）
    # 输入通常是来自编码器的输出 enc_outputs（例如RNN的隐藏状态、Transformer的memory等）。
    # 子类必须重写这个函数，否则会抛出 NotImplementedError。
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


"""
    举例：RNN 解码器子类实现:

    class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    # init_state() 的参数 enc_outputs 是由 编码器的 forward() 输出 传进来的，手动显式写出
    # enc_outputs = encoder(X_src)
    # dec_state = decoder.init_state(enc_outputs)

    def init_state(self, enc_outputs, *args):
    
        # 根据编码器输出初始化解码器状态
        # enc_outputs: (enc_outputs, enc_state)
        # 这里只使用编码器的最终隐藏状态 enc_state
        
        return enc_outputs[1]

    # forward(self, X, state)的state参数以init_state的输出传入

    def forward(self, X, state):
    
        # X: [batch_size, num_steps] —— 解码器输入（训练时通常为上一个真实 token）
        # state: [num_layers, batch_size, num_hiddens] —— 上一个时刻的隐藏状态
        # 返回:
        #     output: [batch_size, num_steps, vocab_size]
        #     state:  [num_layers, batch_size, num_hiddens]
        
        # 1️⃣ 嵌入层：将词索引转换为embedding向量
        X = self.embedding(X).permute(1, 0, 2)  # [num_steps, batch, embed_size]

        # 2️⃣ 为每个时间步扩展编码器最终状态（上下文）
        # state[-1] 是最后一层隐藏状态，用于作为上下文拼接
        context = state[-1].repeat(X.shape[0], 1, 1)  # [num_steps, batch, num_hiddens]
        X_and_context = torch.cat((X, context), 2)     # 拼接到输入： [num_steps, batch, embed+hidden]

        # 3️⃣ GRU 前向传播
        output, state = self.rnn(X_and_context, state)

        # 4️⃣ 输出层，将隐藏状态映射到词表维度
        output = self.dense(output).permute(1, 0, 2)  # [batch, num_steps, vocab_size]
        return output, state

"""

"""
    顺便把seq2seq说清楚：
    https://chatgpt.com/s/t_68e79ef1f12c81918178c27a484d1638
    https://chatgpt.com/s/t_68e79f0ca0fc81918d18d830e01c4a74
    https://chatgpt.com/s/t_68e79fcd93288191997bbace23c1c519
"""

# ========================
# 合并编码器和解码器
# ========================

#@save
# 合并编码器和解码器
# 对着上面encoder和decoder就能看个八九不离十了
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
