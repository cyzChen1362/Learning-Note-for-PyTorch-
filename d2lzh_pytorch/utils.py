import collections
import math
import os
import random
import sys
import tarfile
import time
import json
import zipfile
from tqdm import tqdm
from PIL import Image
from collections import namedtuple
from torch.utils import data

import hashlib
import requests
import pandas as pd
import shutil

from IPython import display
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.ops import box_iou

# import torchtext
# import torchtext.vocab as Vocab
import numpy as np

# ###################### 3.2 ############################
def set_figsize(figsize=(6, 4)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j) 

def linreg(X, w, b):
    return torch.mm(X, w) + b

def squared_loss(y_hat, y): 
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return ((y_hat - y.view(y_hat.size())) ** 2) / 2

def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data



# ######################3##### 3.5 #############################
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    # plt.show()

# 5.6 修改
# def load_data_fashion_mnist(batch_size, root='~/Datasets/FashionMNIST'):
#     """Download the fashion mnist dataset and then load into memory."""
#     transform = transforms.ToTensor()
#     mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
#     mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
#     if sys.platform.startswith('win'):
#         num_workers = 0  # 0表示不用额外的进程来加速读取数据
#     else:
#         num_workers = 4
#     train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     return train_iter, test_iter




# ########################### 3.6  ###############################
# (3.13节修改)
# def evaluate_accuracy(data_iter, net):
#     acc_sum, n = 0.0, 0
#     for X, y in data_iter:
#         acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#         n += y.shape[0]
#     return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到
            
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))




# ########################### 3.7 #####################################3
class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


# ########################### 3.11 ###############################
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(6, 4)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    # plt.show()




# ############################# 3.13 ##############################
# 5.5 修改
# def evaluate_accuracy(data_iter, net):
#     acc_sum, n = 0.0, 0
#     for X, y in data_iter:
#         if isinstance(net, torch.nn.Module):
#             net.eval() # 评估模式, 这会关闭dropout
#             acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#             net.train() # 改回训练模式
#         else: # 自定义的模型
#             if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
#                 # 将is_training设置成False
#                 acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
#             else:
#                 acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
#         n += y.shape[0]
#     return acc_sum / n






# ########################### 5.1 #########################
def corr2d(X, K):  
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y



# ############################ 5.5 #########################

reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)

def accuracy(y_hat, y):
    """计算预测正确的数量

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))



# ########################## 5.6 #########################3
def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter



############################# 5.8 ##############################
class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])



# ########################### 5.11 ################################
class Residual(nn.Module): 
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)
    
def resnet18(output=10, in_channels=3):
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, output))) 
    return net



# ############################## 6.3 ##################################3
def load_data_jay_lyrics():
    """加载周杰伦歌词数据集"""
    with zipfile.ZipFile('../../data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)

def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y





# ###################################### 6.4 ######################################
def one_hot(x, n_class, dtype=torch.float32): 
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res

def to_onehot(X, n_class):  
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else: 
            # 否则需要使用detach函数从计算图分离隐藏状态, 这是为了
            # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                for s in state:
                    s.detach_()
            
            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())
            
            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))

                
                
                
# ################################### 6.5 ################################################
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1) 
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        X = to_onehot(inputs, self.vocab_size) # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                      char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]] # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple): # LSTM, state:(h, c)  
                state = (state[0].to(device), state[1].to(device))
            else:   
                state = state.to(device)
            
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance (state, tuple): # LSTM, state:(h, c)  
                    state = (state[0].detach(), state[1].detach())
                else:   
                    state = state.detach()
    
            (output, state) = model(X, state) # output: 形状为(num_steps * batch_size, vocab_size)
            
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())
            
            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        
        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))




# ######################################## 7.2 ###############################################
def train_2d(trainer):  
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1和s2是自变量状态，本章后续几节会使用
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results

def show_trace_2d(f, results):  
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    #
    plt.show()




# ######################################## 7.3 ###############################################
def get_data_ch7():  
    data = np.genfromtxt('../../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
        torch.tensor(data[:1500, -1], dtype=torch.float32) # 前1500个样本(每个样本5个特征)

def train_ch7(optimizer_fn, states, hyperparams, features, labels,
              batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = linreg, squared_loss
    
    w = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32),
                           requires_grad=True)
    b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)
    
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X, w, b), y).mean()  # 使用平均损失
            
            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
                
            l.backward()
            optimizer_fn([w, b], states, hyperparams)  # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # 每100个样本记录下当前训练误差
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

# 本函数与原书不同的是这里第一个参数优化器函数而不是优化器的名字
# 例如: optimizer_fn=torch.optim.SGD, optimizer_hyperparams={"lr": 0.05}
def train_pytorch_ch7(optimizer_fn, optimizer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    # 初始化模型
    net = nn.Sequential(
        nn.Linear(features.shape[-1], 1)
    )
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            # 除以2是为了和train_ch7保持一致, 因为squared_loss中除了2
            l = loss(net(X).view(-1), y) / 2 
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #
    plt.show()




############################## 8.3 ##################################
class Benchmark():
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))





# ########################### 9.1 ########################################
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    # plt.show()
    return axes

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y) 
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))




############################## 9.3 #####################
def bbox_to_rect(bbox, color):
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)




############################ 9.4 ###########################
def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        feature_map: torch tensor, Shape: [N, C, H, W].
        sizes: List of sizes (0~1) of generated MultiBoxPriores. 
        ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores. 
    Returns:
        anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1
    """
    pairs = [] # pair of (size, sqrt(ration))
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])
    
    pairs = np.array(pairs)
    
    ss1 = pairs[:, 0] * pairs[:, 1] # size * sqrt(ration)
    ss2 = pairs[:, 0] / pairs[:, 1] # size / sqrt(ration)
    
    base_anchors = np.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2
    
    h, w = feature_map.shape[-2:]
    shifts_x = np.arange(0, w) / w
    shifts_y = np.arange(0, h) / h
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
    
    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))
    
    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)

# 这里我直接把MultiBoxPrior换成新的GPU版

def MultiBoxPrior_GPU(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    r"""
    按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    跟原版除了用GPU和tensor规避numpy之外基本没有不同
    Args:
        feature_map: torch tensor, Shape: [N, C, H, W].
        sizes: List of sizes (0~1) of generated MultiBoxPriores.
        ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores.
    Returns:
        anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1
    """
    device = feature_map.device
    dtype  = feature_map.dtype

    pairs = [] # pair of (size, sqrt(ration))
    # 只对包含s1或r1的大小与宽高比的组合感兴趣
    for r in ratios:
        pairs.append((sizes[0], math.sqrt(r)))
    for s in sizes[1:]:
        pairs.append((s, math.sqrt(ratios[0])))

    # 将嵌套列表转为torch.tensor，才有“所有行的第一列”
    # 总长度为n+m-1
    pairs = torch.tensor(pairs, dtype=dtype, device=device)      # (n+m-1, 2)

    # 宽系数
    ss1 = pairs[:, 0] * pairs[:, 1]                              # size * sqrt(r)
    # 高系数
    ss2 = pairs[:, 0] / pairs[:, 1]                              # size / sqrt(r)

    # 生成形如 [xmin, ymin, xmax, ymax] 的二维数组，n+m-1行，4列
    # 这些坐标是相对于中心点的半宽半高，宽系数 0.5 → 左右坐标 ±0.25
    base_anchors = torch.stack([-ss1, -ss2, ss1, ss2], dim=1)    # (k, 4)
    base_anchors = base_anchors / 2.0

    # 取出倒数两个维度的大小
    h, w = feature_map.shape[-2:]
    # 横向锚框中心点坐标(归一化到0~1)
    shifts_x = torch.arange(w, dtype=dtype, device=device) / w
    # 纵向锚框中心点坐标(归一化到0~1)
    shifts_y = torch.arange(h, dtype=dtype, device=device) / h

    # shift_x：每一行都复制 shifts_x，表示每个网格点的横坐标
    # [[0, 1/w, 2/w, ...],
    #  [0, 1/w, 2/w, ...],
    #  ...
    # ]
    # shift_y：每一列都复制 shifts_y，表示每个网格点的纵坐标
    # [[0, 0, 0, ...],
    #  [1/h, 1/h, 1/h, ...],
    #  ...
    # ]
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")  # (h,w)

    # 拉平成一维，[h*w, 4] -> (x, y, x, y)
    # 然后把 (x, y) 复制成 [x, y, x, y]
    # → 行
    # ↓ 列  [[0  , 1/w, 2/w, ... , 0  , 1/w, 2/w, ...],
    #        [0  , 0  , 0  , ... , 1/h, 1/h, 1/h, ...],
    #        [0  , 1/w, 2/w, ... , 0  , 1/w, 2/w, ...],
    #        [0  , 0  , 0  , ... , 1/h, 1/h, 1/h, ...],
    #       ]
    shifts = torch.stack([
        shift_x.reshape(-1),
        shift_y.reshape(-1),
        shift_x.reshape(-1),
        shift_y.reshape(-1)
    ], dim=1)

    # 广播相加得到最终 anchors
    # 等价于 anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))
    # shifts.reshape：行数仍然是h*w，第三维仍然是4，新增了1个第二维
    # 也就是shifts.reshape是(h*w,1,4)，base_anchors.reshape是(1,n+m-1,4)
    # 就能广播机制了
    # 生成的anchors是(h*w,n+m-1,4)
    # anchors的每一行都对应着同一个中心坐标，
    # 这一行的每一列都代表着一种不同的锚框大小，
    # 第三维4个通道说明每个中心坐标的每种锚框都由四个坐标点构成
    # 当然以上全是归一化的形式
    # 以及，这里当然是加法而不是惩罚（没错，就是在你认为的广播机制条件下）
    # 因为shifts的某一行是一个特征图上某个像素位置对应到原图的归一化中心点坐标 [cx, cy, cx, cy]
    # base_anchors的每一行则是某个归一化中心点的归一化偏移量
    # 总偏移后坐标 = （宽or高） * （归一化中心点 + 归一化偏移量）
    # 或者更形象来说，这里计算的偏移量是s√r和s/√r，而总偏移量是ws√r和hs/√r
    # 刚好总中心点是wcx和hcy，这里的shifts和base_anchors都提取出来了w和h项

    anchors = shifts[:, None, :] + base_anchors[None, :, :]      # (h*w, k, 4)
    anchors = anchors.reshape(1, -1, 4)                          # (1, h*w*k, 4)
    return anchors

def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().cpu().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=6, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

def compute_intersection(set_1, set_2):
    """
    计算anchor之间的交集
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def compute_jaccard(set_1, set_2):
    """
    计算anchor之间的Jaccard系数(IoU)
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # Find intersections
    intersection = compute_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def assign_anchor(bb, anchor, jaccard_threshold=0.5):
    """
    # 按照「9.4.1. 生成多个锚框」图9.3所讲为每个anchor分配真实的bb, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        bb: 真实边界框(bounding box), shape:（nb, 4）
        anchor: 待分配的anchor, shape:（na, 4）
        jaccard_threshold: 预先设定的阈值
    Returns:
        assigned_idx: shape: (na, ), 每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
    """
    device = anchor.device
    na = anchor.shape[0]
    nb = bb.shape[0]

    # 这一部分为了规避cpu，直接在gpu上计算
    # 源代码
    # jaccard = compute_jaccard(anchor, bb).detach().cpu().numpy() # shape: (na, nb)
    # 新代码
    jaccard = compute_jaccard(anchor, bb)  # (na, nb)

    assigned_idx = torch.full((na,), -1, dtype=torch.long, device=device)  # 初始全为-1
    
    # 先为每个bb分配一个anchor(不要求满足jaccard_threshold)
    # 这里因为改用Tensor，所以用.clone()而不是.copy()
    # 源代码
    # jaccard_cp = jaccard.copy()
    # 新代码
    jaccard_cp = jaccard.clone().to(jaccard.device)

    for j in range(nb):
        # 源代码
        # i = np.argmax(jaccard_cp[:, j])
        # 新代码
        i = torch.argmax(jaccard_cp[:, j]).item()
        assigned_idx[i] = j
        jaccard_cp[i, :] = float("-inf") # 赋值为负无穷, 相当于去掉这一行
     
    # 处理还未被分配的anchor, 要求满足jaccard_threshold
    for i in range(na):
        if assigned_idx[i] == -1:
            # 源代码
            # j = np.argmax(jaccard[i, :])
            # 新代码
            j = torch.argmax(jaccard[i, :]).item()
            if jaccard[i, j] >= jaccard_threshold:
                assigned_idx[i] = j

    # 保证返回的张量与 anchor 在同一设备
    return assigned_idx

def xy_to_cxcy(xy):
    """
    将(x_min, y_min, x_max, y_max)形式的anchor转换成(center_x, center_y, w, h)形式的.
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    Args:
        xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    Returns: 
        bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h

# 这里我直接写原版的坐标变换了，更显式而且实现功能完全相同

def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）

    Defined in :numref:`sec_bbox`"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）

    Defined in :numref:`sec_bbox`"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def MultiBoxTarget(anchor, label):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        anchor: torch tensor, 输入的锚框, 一般是通过MultiBoxPrior生成, shape:（1，锚框总数，4）
        label: 真实标签, shape为(bn, 每张图片最多的真实锚框数, 5)
               第二维中，如果给定图片没有这么多锚框, 可以先用-1填充空白, 最后一维中的元素为[类别标签, 四个坐标值]
    Returns:
        列表, [bbox_offset, bbox_mask, cls_labels]
        bbox_offset: 每个锚框的标注偏移量，形状为(bn，锚框总数*4)
        bbox_mask: 形状同bbox_offset, 每个锚框的掩码, 一一对应上面的偏移量, 负类锚框(背景)对应的掩码均为0, 正类锚框的掩码均为1
        cls_labels: 每个锚框的标注类别, 其中0表示为背景, 形状为(bn，锚框总数)
    """
    # print("anchor device:", anchor.device, "label device:", label.device)

    # 这里兼容旧版：旧版是无真实框行全为0，然后算IoU时必然为0，相当于跳过了
    # 新版这里应该也可以兼容旧数据集......所以就不改了

    assert len(anchor.shape) == 3 and len(label.shape) == 3
    bn = label.shape[0]
    
    def MultiBoxTarget_one(anc, lab, eps=1e-6):
        """
        MultiBoxTarget函数的辅助函数, 处理batch中的一个
        Args:
            anc: shape of (锚框总数, 4)
            lab: shape of (真实锚框数, 5), 5代表[类别标签, 四个坐标值]
            eps: 一个极小值, 防止log0
        Returns:
            offset: (锚框总数*4, )
            bbox_mask: (锚框总数*4, ), 0代表背景, 1代表非背景
            cls_labels: (锚框总数, 4), 0代表背景
        """
        an = anc.shape[0]
        # 上一级assign_anchor我也改了device
        assigned_idx = assign_anchor(lab[:, 1:], anc) # (锚框总数, )
        bbox_mask = ((assigned_idx >= 0).float().unsqueeze(-1)).repeat(1, 4) # (锚框总数, 4)

        # 然后关于device不同导致出错的问题，这里也修改了一下

        device = anc.device
        cls_labels = torch.zeros(an, dtype=torch.long, device=device) # 0表示背景
        assigned_bb = torch.zeros((an, 4), dtype=torch.float32, device=device) # 所有anchor对应的bb坐标
        for i in range(an):
            bb_idx = assigned_idx[i]
            if bb_idx >= 0: # 即非背景
                cls_labels[i] = lab[bb_idx, 0].long().item() + 1 # 注意要加一
                assigned_bb[i, :] = lab[bb_idx, 1:]

        center_anc = xy_to_cxcy(anc) # (center_x, center_y, w, h)
        center_assigned_bb = xy_to_cxcy(assigned_bb)

        offset_xy = 10.0 * (center_assigned_bb[:, :2] - center_anc[:, :2]) / center_anc[:, 2:]
        offset_wh = 5.0 * torch.log(eps + center_assigned_bb[:, 2:] / center_anc[:, 2:])
        offset = torch.cat([offset_xy, offset_wh], dim = 1) * bbox_mask # (锚框总数, 4)

        return offset.view(-1), bbox_mask.view(-1), cls_labels
    
    batch_offset = []
    batch_mask = []
    batch_cls_labels = []
    for b in range(bn):
        offset, bbox_mask, cls_labels = MultiBoxTarget_one(anchor[0, :, :], label[b, :, :])
        
        batch_offset.append(offset)
        batch_mask.append(bbox_mask)
        batch_cls_labels.append(cls_labels)
    
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    cls_labels = torch.stack(batch_cls_labels)
    
    return [bbox_offset, bbox_mask, cls_labels]

@torch.no_grad()
def MultiBoxTarget_fast(anchor, label, jaccard_threshold=0.5, eps=1e-6):
    """
    更快的 MultiBoxTarget_fast 实现：
    使用 torchvision.ops.box_iou 替代 Python 循环，大幅减少 CPU 端开销。
    其余没变，只是改了一下代码写法

    Args:
        anchor: torch tensor, 输入的锚框, 一般是通过MultiBoxPrior生成, shape:（1，锚框总数，4）
        label: 真实标签, shape为(bn, 每张图片最多的真实锚框数, 5)
               第二维中，如果给定图片没有这么多锚框, 可以先用-1填充空白, 最后一维中的元素为[类别标签, 四个坐标值]
    Returns:
        列表, [bbox_offset, bbox_mask, cls_labels]
        bbox_offset: 每个锚框的标注偏移量，形状为(bn，锚框总数*4)
        bbox_mask: 形状同bbox_offset, 每个锚框的掩码, 一一对应上面的偏移量, 负类锚框(背景)对应的掩码均为0, 正类锚框的掩码均为1
        cls_labels: 每个锚框的标注类别, 其中0表示为背景, 形状为(bn，锚框总数)
    """
    assert anchor.ndim == 3 and label.ndim == 3
    device = anchor.device
    A = anchor.size(1)              # 锚框数
    anc = anchor[0]                 # 所有锚框的坐标(A, 4)
    center_anc = xy_to_cxcy(anc)

    batch_offsets, batch_masks, batch_cls = [], [], []
    B = label.size(0)

    for b in range(B):
        lab = label[b]
        # 过滤掉无效行（类别<0）
        valid = lab[:, 0] >= 0
        lab = lab[valid]
        if lab.numel() == 0:
            # 没有真实框：全部背景
            batch_offsets.append(torch.zeros(A * 4, device=device))
            batch_masks.append(torch.zeros(A * 4, device=device))
            batch_cls.append(torch.zeros(A, dtype=torch.long, device=device))
            continue

        gt_cls = lab[:, 0].long() + 1               # 正类从1开始
        gt_box = lab[:, 1:5]  # 真实框坐标

        # 其实这段代码和新书代码的逻辑是完全一样的，只是有点绕

        # 新书的逻辑是先为box分配anchor，
        # 然后剩余的anchor，看看和它最匹配的box是否符合阈值，符合就分配

        # 原书新代码的逻辑是先为anchor分配box，同时要符合阈值
        # 然后剩余的box再去分配anchor，并且可能覆盖之前的结果

        # 也就是说，这两种代码的逻辑都是box分配anchor的优先级最高
        # 然后anchor分配box的优先级次之，并且要符合逻辑
        # 新书的代码是在逻辑上确定优先级，旧书的新代码是以之后的结果能覆盖之前的结果确定优先级

        # IoU 计算 (A, G)，返回[N, M] 的张量，
        # 第 i 行、第 j 列元素表示 boxes1[i] 与 boxes2[j] 的 IoU
        ious = box_iou(anc, gt_box)

        # 先每个 anchor 找 IoU 最大的 GT
        # best_ious：[A] 每个锚框与所有真实框的最大 IoU。
        # best_gt：  [A] 给出最大 IoU 对应的真实框索引。
        best_ious, best_gt = ious.max(dim=1)
        assigned_idx = torch.full((A,), -1, dtype=torch.long, device=device)
        # 生成布尔张量 [A]，对于 IoU 大于等于阈值的锚框，对应位置为 True（认为是正样本）
        pos = best_ious >= jaccard_threshold
        # 把正样本锚框的位置替换为它们匹配到的真实框索引
        assigned_idx[pos] = best_gt[pos]

        # 保证每个 GT 至少分到一个 anchor
        best_anchor = ious.argmax(dim=0)
        # 例如这里bast_anchor是[2 0 1]，然后torch.arange返回是[0 1 2]
        # 那么就是assigned_idx[2]=0, assigned_idx[0]=1, assigned_idx[1]=2
        assigned_idx[best_anchor] = torch.arange(gt_box.size(0), device=device)

        # 构造标签和偏移
        cls_labels = torch.zeros(A, dtype=torch.long, device=device)
        assigned_bb = torch.zeros(A, 4, device=device)
        # 这里的pos是长为A的布尔变量
        pos = assigned_idx >= 0
        if pos.any():
            cls_labels[pos] = gt_cls[assigned_idx[pos]]
            assigned_bb[pos] = gt_box[assigned_idx[pos]]

        center_gt = xy_to_cxcy(assigned_bb)
        # 这里的公式见书本
        # 当然，书本是写了四项，这里是x y 两项算一次，w h 两项算一次，所以只有两条式子
        # 这里算x y偏移量
        offset_xy = 10.0 * (center_gt[:, :2] - center_anc[:, :2]) / (center_anc[:, 2:] + eps)
        # 同理，这里算w h偏移量，除以0.1标准差就是乘以10
        offset_wh =  5.0 * torch.log((center_gt[:, 2:] / (center_anc[:, 2:] + eps)).clamp(min=eps))
        # 拼起来；如果都没有分配锚框那就没有偏移量，掩码覆盖为0
        offsets = torch.cat([offset_xy, offset_wh], dim=1)
        # -1 表示保持原来的 N 不变，4 表示复制到 4 列
        mask = pos.unsqueeze(1).expand(-1, 4).float()

        batch_offsets.append((offsets * mask).reshape(-1))
        batch_masks.append(mask.reshape(-1))
        batch_cls.append(cls_labels)

    # 原先是一个Python list
    # 现在在新的第一维把这些小张量“堆叠”起来，形成(bn, N*4)的单个 torch.Tensor
    bbox_offset = torch.stack(batch_offsets)
    bbox_mask   = torch.stack(batch_masks)
    cls_labels  = torch.stack(batch_cls)
    return [bbox_offset, bbox_mask, cls_labels]


Pred_BB_Info = namedtuple("Pred_BB_Info", ["index", "class_id", "confidence", "xyxy"])
def non_max_suppression(bb_info_list, nms_threshold = 0.5):
    """
    非极大抑制处理预测的边界框
    Args:
        bb_info_list: Pred_BB_Info的列表, 包含预测类别、置信度等信息
        nms_threshold: 阈值
    Returns:
        output: Pred_BB_Info的列表, 只保留过滤后的边界框信息
    """
    output = []
    # 先根据置信度从高到低排序
    sorted_bb_info_list = sorted(bb_info_list, key = lambda x: x.confidence, reverse=True)

    while len(sorted_bb_info_list) != 0:
        best = sorted_bb_info_list.pop(0)
        output.append(best)
        
        if len(sorted_bb_info_list) == 0:
            break

        bb_xyxy = []
        for bb in sorted_bb_info_list:
            bb_xyxy.append(bb.xyxy)
        
        iou = compute_jaccard(torch.tensor([best.xyxy]), 
                              torch.tensor(bb_xyxy))[0] # shape: (len(sorted_bb_info_list), )
        
        n = len(sorted_bb_info_list)
        sorted_bb_info_list = [sorted_bb_info_list[i] for i in range(n) if iou[i] <= nms_threshold]
    return output

def MultiBoxDetection(cls_prob, loc_pred, anchor, nms_threshold = 0.5):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        cls_prob: 经过softmax后得到的各个锚框的预测概率, shape:(bn, 预测总类别数+1, 锚框个数)
        loc_pred: 预测的各个锚框的偏移量, shape:(bn, 锚框个数*4)
        anchor: MultiBoxPrior输出的默认锚框, shape: (1, 锚框个数, 4)
        nms_threshold: 非极大抑制中的阈值
    Returns:
        所有锚框的信息, shape: (bn, 锚框个数, 6)
        每个锚框信息由[class_id, confidence, xmin, ymin, xmax, ymax]表示
        class_id=-1 表示背景或在非极大值抑制中被移除了
    """
    # ********************************************************************************
    # 新书代码上其实有些坑，例如MultiBoxDetection偏移量编码用了log解码却没exp
    # 所以后面multibox_detection用原书的
    # 新书的MultiBoxDetection可以用作理解
    # ********************************************************************************

    assert len(cls_prob.shape) == 3 and len(loc_pred.shape) == 2 and len(anchor.shape) == 3
    bn = cls_prob.shape[0]
    
    def MultiBoxDetection_one(c_p, l_p, anc, nms_threshold = 0.5):
        """
        MultiBoxDetection的辅助函数, 处理batch中的一个
        Args:
            c_p: (预测总类别数+1, 锚框个数)
            l_p: (锚框个数*4, )
            anc: (锚框个数, 4)
            nms_threshold: 非极大抑制中的阈值
        Return:
            output: (锚框个数, 6)
        """
        pred_bb_num = c_p.shape[1]
        anc = (anc + l_p.view(pred_bb_num, 4)).detach().cpu().numpy() # 加上偏移量
        
        confidence, class_id = torch.max(c_p, 0)
        confidence = confidence.detach().cpu().numpy()
        class_id = class_id.detach().cpu().numpy()
        
        pred_bb_info = [Pred_BB_Info(
                            index = i,
                            class_id = class_id[i] - 1, # 正类label从0开始
                            confidence = confidence[i],
                            xyxy=[*anc[i]]) # xyxy是个列表
                        for i in range(pred_bb_num)]
        
        # 正类的index
        obj_bb_idx = [bb.index for bb in non_max_suppression(pred_bb_info, nms_threshold)]
        
        output = []
        for bb in pred_bb_info:
            output.append([
                (bb.class_id if bb.index in obj_bb_idx else -1.0),
                bb.confidence,
                *bb.xyxy
            ])
            
        return torch.tensor(output) # shape: (锚框个数, 6)
    
    batch_output = []
    for b in range(bn):
        batch_output.append(MultiBoxDetection_one(cls_prob[b], loc_pred[b], anchor[0], nms_threshold))
    
    return torch.stack(batch_output)



# ################################# 9.6 ############################
class PikachuDetDataset(torch.utils.data.Dataset):
    """皮卡丘检测数据集类"""
    def __init__(self, data_dir, part, image_size=(256, 256)):
        assert part in ["train", "val"]
        self.image_size = image_size
        self.image_dir = os.path.join(data_dir, part, "images")
        
        with open(os.path.join(data_dir, part, "label.json")) as f:
            self.label = json.load(f)
            
        self.transform = torchvision.transforms.Compose([
            # 将 PIL 图片转换成位于[0.0, 1.0]的floatTensor, shape (C x H x W)
            torchvision.transforms.ToTensor()])
            
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        image_path = str(index + 1) + ".png"
        
        cls = self.label[image_path]["class"]
        label = np.array([cls] + self.label[image_path]["loc"], 
                         dtype="float32")[None, :]
        
        PIL_img = Image.open(os.path.join(self.image_dir, image_path)
                            ).convert('RGB').resize(self.image_size)
        img = self.transform(PIL_img)
        
        sample = {
            "label": label, # shape: (1, 5) [class, xmin, ymin, xmax, ymax]
            "image": img    # shape: (3, *image_size)
        }
        
        return sample

def load_data_pikachu(batch_size, edge_size=256, data_dir = '../../data/pikachu'):  
    """edge_size：输出图像的宽和高"""
    image_size = (edge_size, edge_size)
    train_dataset = PikachuDetDataset(data_dir, 'train', image_size)
    val_dataset = PikachuDetDataset(data_dir, 'val', image_size)
    

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                             shuffle=True, num_workers=4)

    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=4)
    return train_iter, val_iter

# ################################# 9.6.5 ############################

# 由于新书9.6-9.13仍有部分未施工完毕
# 所以9.6-9.13将会跟随原书学习，同时修改部分d2l代码
# 使用香蕉数据集，契合原书9.6-9.13

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

DATA_HUB['banana-detection'] = (
    DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

DATA_HUB['voc2012'] = (DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名

    Defined in :numref:`sec_kaggle_house`"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """下载并解压zip/tar文件

    Defined in :numref:`sec_kaggle_house`"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def read_data_bananas(is_train=True):
    # 先按照 d2l 默认规则下载
    tmp_dir = download_extract('banana-detection')

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

    # targets转成torch.Tensor，形状为[N, 5] [batch, label xmin ymin xmax ymax]
    # 将坐标除以 256，实现归一化到 0–1 区间(数据集中图像尺寸是 256×256)
    return images, torch.tensor(targets).unsqueeze(1) / 256

class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集

    Defined in :numref:`sec_object-detection-dataset`"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    """加载香蕉检测数据集

    Defined in :numref:`sec_object-detection-dataset`"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True, num_workers=0, pin_memory=True)
                                            # , persistent_workers=True
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size, num_workers=0, pin_memory=True)
                                            # , persistent_workers=True
    return train_iter, val_iter

# ################################# 9.7 #########################

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(7, 4)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

        # 👇 强制 y 轴更多刻度并设置显示格式
        ax = self.axes[0]
        ax.yaxis.set_major_locator(MultipleLocator(0.05))  # 0,0.05,0.10,0.15,0.20
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.fig.tight_layout()

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x_i, y_i, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_i, y_i, fmt, linewidth=1.5)
        self.config_axes()
        # 追加一次，确保每次刷新后刻度依旧
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter
        ax = self.axes[0]
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.fig.tight_layout()
        display.display(self.fig)
        # display.clear_output(wait=True)

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框

    Defined in :numref:`subsec_labeling-anchor-boxes`"""
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序
        Defined in :numref:`subsec_predicting-bounding-boxes-nms`
        Args:
            boxes: 各预测框的坐标
            scores: 各个框的预测概率（对了，交并比可不是预测概率）
            iou_threshold: 顾名思义
        Returns:
            所有锚框的信息, shape: (bn, 锚框个数, 6)
        """
    # 按置信度从高到低排序
    B = torch.argsort(scores, dim=-1, descending=True) # 类似于tensor([2, 0, 1, ...])
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        # 最高分框索引
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        # 这里并没有最高分框和自己IoU哈，看清楚一点代码
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        # 这里iou的第0个元素是原本B的第1个元素了，所以后面才会+1
        # torch.nonzero(...)：[True, False, True] → tensor([[0],[2]])
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """
    使用非极大值抑制来预测边界框
    注：这个代码其实并没有真正区分多类别，原书也没有真正区分
    可能后面多类别分类任务会先分开类别再搞，但我还是喜欢改detection代码......
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
        https://zh.d2l.ai/chapter_computer-vision/anchor.html
        Args:
            cls_prob: 经过softmax后得到的各个锚框的预测概率, shape:[N, total_anchors, num_classes+1]
            offset_preds: 预测的各个锚框的偏移量, shape:[N, total_anchors * 4]，回忆一下书本里的偏移量，就是4个参数
            anchor: MultiBoxPrior输出的默认锚框, shape: [1, total_anchors, 4]
            nms_threshold: 非极大抑制中的阈值
            pos_threshold: 非背景预测的阈值
        Returns:
            output(batch_size, num_anchors, 6)

        """
    # 非极大值抑制的原理看新书

    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        # cls_prob:[total_anchors, num_classes+1]
        # offset_pred:[total_anchors, 4]
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        # 在第 0 维度上取最大，也就是在所有类别（不包括背景）之间找一个最大值
        # 如果是torch.max(cls_prob[:], 0)，连背景都包括了，那就是预测背景了
        # conf：最大值； class_id：类别索引（从0开始）
        conf, class_id = torch.max(cls_prob[1:], 0)
        # 预测框：锚框+偏差
        predicted_bb = offset_inverse(anchors, offset_pred)
        # 返回一列索引序列，类似于[2, 1, 6, ...]
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        # unique(return_counts=True)：
        # uniques：所有索引不重复地列一遍
        # counts ：每个索引出现的次数
        uniques, counts = combined.unique(return_counts=True)
        # 只出现了一次的必然是被NMS筛下去的锚框
        non_keep = uniques[counts == 1]
        # 先保留框，再接上被去掉的框
        all_id_sorted = torch.cat((keep, non_keep))
        # 被NMS筛下去的锚框分类为背景
        class_id[non_keep] = -1
        # 重新排列，保证一致
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        # 主要是某些确实通过NMS了，但conf很低的，也去掉吧，返回布尔列向量
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        # 既然某类的conf低，反过来就是背景的conf高
        conf[below_min_idx] = 1 - conf[below_min_idx]
        # pred_info：[num_anchors, 6]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        # 按每张图片append
        out.append(pred_info)
    return torch.stack(out)

# ################################# 9.9 #########################
def read_voc_images_old(root="../../data/VOCdevkit/VOC2012",
                    is_train=True, max_num=None):
    txt_fname = '%s/ImageSets/Segmentation/%s' % (
        root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    if max_num is not None:
        images = images[:min(max_num, len(images))]
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in tqdm(enumerate(images)):
        features[i] = Image.open('%s/JPEGImages/%s.jpg' % (root, fname)).convert("RGB")
        labels[i] = Image.open('%s/SegmentationClass/%s.png' % (root, fname)).convert("RGB")
    return features, labels # PIL image

# colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
# for i, colormap in enumerate(VOC_COLORMAP):
#     colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
def voc_label_indices_old(colormap, colormap2label):
    """
    convert colormap (PIL image) to colormap2label (uint8 tensor).
    """
    colormap = np.array(colormap.convert("RGB")).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def voc_rand_crop_old(feature, label, height, width):
    """
    Random crop feature (PIL image) and label (PIL image).
    """
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            feature, output_size=(height, width))
    
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)    

    return feature, label

class VOCSegDataset_old(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir, colormap2label, max_num=None):
        """
        crop_size: (h, w)
        """
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean, 
                                             std=self.rgb_std)
        ])
        
        self.crop_size = crop_size # (h, w)
        features, labels = read_voc_images_old(root=voc_dir,
                                           is_train=is_train, 
                                           max_num=max_num)
        self.features = self.filter(features) # PIL image
        self.labels = self.filter(labels)     # PIL image
        self.colormap2label = colormap2label
        print('read ' + str(len(self.features)) + ' valid examples')

    def filter(self, imgs):
        return [img for img in imgs if (
            img.size[1] >= self.crop_size[0] and
            img.size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop_old(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        
        return (self.tsf(feature),
                voc_label_indices_old(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

# ############################# 9.9.5 ##########################
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
        tmp_dir = download_extract('VOC2012')  # 只在需要时下载
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

# VOC颜色映射表
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

# VOC类别名称列表
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

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

# 随机裁剪特征(feature)和标签(label)图像
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

# 自定义语义分割数据集类
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


# 整合所有组件，写成VOC数据集下载函数
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
        voc_dir = download_extract('voc2012', os.path.join('VOCdevkit', 'VOC2012'))
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

# ############################# 9.11 ##########################

def train_batch_ch13(net, X, y, loss, trainer, device):
    """用多GPU进行小批量训练
        其实这里改了单gpu因为我没有多gpu

    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # 微调BERT中所需
        X = [x.to(device) for x in X]
    else:
        X = X.to(device)
    y = y.to(device)

    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()

    # 这里返回的loss是sum起来的，所以后面会有一个metric[0] / metric[2]
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               device):
    """用单一GPU进行模型训练
        因为老子没有多GPU

    Defined in :numref:`sec_image_augmentation`"""
    # 使用 time 记录总时长
    total_time = 0.0

    # 记录曲线
    epochs, train_loss_list, train_acc_list, test_acc_list = [], [], [], []

    # 直接把模型搬到单个 device
    net = net.to(device)

    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            start_t = time.perf_counter()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, device)
            metric.add(l, acc, labels.shape[0], labels.numel())
            total_time += time.perf_counter() - start_t

        test_acc = evaluate_accuracy_gpu(net, test_iter, device)

        # 记录本轮结果
        epochs.append(epoch + 1)
        train_loss_list.append(metric[0] / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_acc)

        print(f'Epoch {epoch + 1}: '
              f'loss {metric[0] / metric[2]:.3f}, '
              f'train acc {metric[1] / metric[3]:.3f}, '
              f'test acc {test_acc:.3f}')

    print(f'{metric[2] * num_epochs / total_time:.1f} examples/sec on {device}')

    # ===== 用 matplotlib 画图 =====
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, train_acc_list, label='Train Acc')
    plt.plot(epochs, test_acc_list, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Curve')
    plt.xlim(1, num_epochs)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

# ############################# 9.13 ##########################
DATA_HUB['cifar10_tiny'] = (DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# 读取trainLabels.csv文件，其格式为一列图片名和一列标签
def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

# 复制文件函数，如果有就不复制了
def copyfile(filename, target_dir):
    """将文件复制到目标目录，如果已存在则跳过"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, os.path.basename(filename))
    if not os.path.exists(target_path):   # 如果不存在再复制
        shutil.copy(filename, target_path)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # ********************************************************************************
    # 最终目录结构形如：
    # train_valid_test/
    #     train_valid/
    #         cat/
    #         dog/
    #     train/
    #         cat/
    #         dog/
    #     valid/
    #         cat/
    #         dog/
    # ********************************************************************************
    # 训练数据集中样本最少的类别中的样本数
    # collections.Counter(labels.values()) 会统计每个类别的样本数量；
    # .most_common() 按数量从多到少排序
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数，即计算每类验证样本的数量
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    # 用一个字典 label_count 记录每个类别目前已分到验证集的数量
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        # 用 train_file.split('.')[0] 去掉文件后缀，比如 cat1.jpg → cat1
        # 再查 labels 获取类别名
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        # 首先把样本复制到一个“汇总文件夹” train_valid/label/ 下
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            # 取出键的值，如果没有这个键就是0
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label

#@save
def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    # ********************************************************************************
    # 执行完函数后，数据集结构变成：
    # train_valid_test/
    #     train/
    #         cat/
    #         dog/
    #     valid/
    #         cat/
    #         dog/
    #     test/
    #         unknown/
    #             img1.jpg
    #             img2.jpg
    #             ...
    # ********************************************************************************
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))

# 整理并读取cifar10数据集
def reorg_cifar10_data(data_dir, valid_ratio = 0.1):
    # 读取标签
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    # 将验证集从原始的训练集中拆分出来
    reorg_train_valid(data_dir, labels, valid_ratio)
    # 在预测期间整理测试集，以方便读取
    reorg_test(data_dir)

# ############################# 10.1 ##########################

def count_corpus(tokens):
    """统计词元的频率

    Defined in :numref:`sec_text_preprocessing`"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # 词频统计
        # 按出现频率排序
        # 调用 count_corpus() 统计每个词出现的次数；
        # 然后按出现频率降序排列，得到形如：[('you', 2), ('love', 2), ('i', 1), ('me', 1)]
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)

        # 建立索引→词的列表idx_to_token 和 词→索引的字典token_to_idx

        # 首先初始化特殊词元
        # 未知词元的索引为0
        # 示例：
        # reserved_tokens = ['<pad>', '<bos>', '<eos>']
        # idx_to_token = ['<unk>', '<pad>', '<bos>', '<eos>']
        # token_to_idx = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

        # 其次添加普通词元
        # 逐个扫描词频表；
        # 只保留出现次数 ≥ min_freq 的词；
        # 避免重复（已存在的词不再添加）
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    # 词表的总大小
    def __len__(self):
        return len(self.idx_to_token)

    # 输入可以是单个 token 或 token 列表；
    # 查不到的词返回 <unk> 的索引（即 0）；
    # 例如：
    # vocab['love'] → 5
    # vocab[['i', 'love', 'you']] → [3, 5, 4]
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    # 反向映射：索引 → 词
    # 例如：
    # vocab.to_tokens([3, 5, 4]) → ['i', 'love', 'you']
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    # unk 属性
    # 指定 <unk> 的索引永远是 0
    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    # token_freqs 属性
    # 可查看词频统计表，用于分析词分布或画 Zipf 定律图
    @property
    def token_freqs(self):
        return self._token_freqs

DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

# 下载数据集
def read_data_nmt():
    """载入“英语－法语”数据集"""
    tmp_dir = download_extract('fra-eng')

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

# 预处理“英语－法语”数据集
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

# 绘制列表长度对的直方图
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    set_figsize(figsize=(8, 6))
    _, _, patches = plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    plt.legend(legend)
    plt.show()

# 截断或填充文本序列
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    # 把输入的 token 序列 line 处理成固定长度 num_steps；
    # 太长就截断；
    # 太短就用 padding_token 填充。
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

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

# 返回翻译数据集的迭代器和词表
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
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 得到目标语料的词表
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])

    # 将源语料的每个句子转成有效长度的张量，以及每句的有效长度
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    # 将目标语料的每个句子转成有效长度的张量，以及每句的有效长度
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)

    # 拼接四个张量
    # 这里data_arrays[1]和data_arrays[3]就是每一句的有效长度，迭代器也会给出
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    # 封装成批量数据迭代器
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


# ############################# 10.2 ##########################

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

# ############################# 10.3 ##########################

# Seq2Seq编码器
class Seq2SeqEncoder(Encoder):
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

# Seq2Seq解码器
class Seq2SeqDecoder(Decoder):
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
        metric = Accumulator(2)  # 训练损失总和，词元数量

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
            grad_clipping(net.parameters(), 1, device)
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

# Seq2Seq预测
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
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
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

# BLEU评估
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

# ############################# 10.4 ##########################

# 注意力权重热力图可视化
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(6, 5),
                  cmap='Reds'):
    """显示矩阵热图"""
    # 输入matrices的形状是 （要显示的行数，要显示的列数，查询的数目，键的数目）
    # 具体每个参数的解析如下：
    # 可以先参考Chapter10_Class6第三部分的注释，理解Q K V先
    # https://chatgpt.com/s/t_68ea1d737348819181ad6c85e39397bd
    # https://chatgpt.com/s/t_68ea1d818bdc8191863ad1dd622353dd
    # https://chatgpt.com/s/t_68ea1d90daf4819181a2c74f794e1336
    use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
    plt.show()

# ############################# 10.5 ##########################

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(6, 4), axes=None):
    """绘制数据点

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

# ############################# 10.6 ##########################

# 掩蔽softmax操作
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    # 例如输入X为(batch_size, num_queries, num_keys)
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        # 如果valid_lens为(batch_size,)
        if valid_lens.dim() == 1:
            # 那么复制valid_lens为(batch_size*num_queries,)
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # 否则直接将valid_lens变为一维(batch_size*num_queries,)
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        # 然后将遮蔽完的X给reshape回去，并沿最后一维num_keys进行softmax
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# 加性注意力
# a(q,k)=Wv.T * tanh(Wq * q + Wk * k)，当然实现的时候直接套模块就可以了
# 注意不要混淆num_queries和query_size
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):

        # queries：(batch_size, num_queries, query_size)
        # num_queries相当于目标句子长度（decoder时间步数），query_size相当于每个query的维度
        # keys：(batch_size，num_keys，key_size)
        # num_keys相当于源句子长度（encoder时间步数），key_size相当于每个词经过encoder后得到的语义向量维度
        # values：(batch_size，num_keys，value_size)
        # num_keys相当于源句子中的词数，value_size相当于每个value的向量维度

        # 其中，queries来自 decoder 当前时刻的隐藏状态（或多个时间步的状态）。
        # 每个 query 向量代表“解码器当前要生成某个词时，它在问：我该关注源句子的哪个部分？”

        # Wq * q，queries：(batch_size, num_queries, num_hiddens)
        queries = self.W_q(queries)
        # Wk * k，keys：(batch_size，num_keys，num_hiddens)
        keys = self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，num_queries，1，num_hidden)
        # key的形状：(batch_size，1，num_keys，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，num_queries，num_keys)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，num_keys，value_size)
        # 最终输出：(batch_size,num_queries,value_size)
        return torch.bmm(self.dropout(self.attention_weights), values)

# ********************************************************************************
# 这里讲一下我对 Scaled Dot-Product Attention 的理解：
#
# 向量化版本：
# Q:n*d; K:m*d; V:m*v;
#
# Q的每一行是一个样本，每个样本是d维向量；K的每一行是一个样本，每个样本是d维度向量
# Q*KT也就是Q[1]和K[1]做内积得到Q*KT[1][1]，Q[1]和K[2]做内积得到Q*KT[1][2]
# 也就是Q*KT[i][j]是Q[i]和K[j]的相似度（如果二者正好正交，内积结果一定是0）
# 然后除以√d相当于除以常数，避免过大
# 得到的注意力分数a(Q,K)=Q*KT/√d是n*m，其中a(Q,K)[i][j]可以理解为Q[i]和K[j]的相似度
#
# 然后对a(Q,K)的每一行进行softmax，
# 那么softmax(a(Q,K))[i]也就是对于Q[i]而言，根据和K每个样本的相似度算出来的权重向量
# 例如softmax(a(Q,K))[i][j]就是K[j]对于Q[i]的权重，softmax(a(Q,K))[i][j+1]就是K[j+1]对于Q[i]的权重
#
# 然后对于注意力池化，softmax(a(Q,K))*V
# softmax(a(Q,K))每一行都是权重向量，V的每一列都是对于某个维度，所有样本各自的值
# 那么输出结果的第i个样本的第j个维度的值，肯定是V所有样本在这一维度的值根据权重求和
# 也就是softmax(a(Q,K))的第i行乘以V的第j列
#
# 顺便一提，K和V是键值对，这俩是捆绑在一起的
# 权重矩阵softmax(a(Q,K))每一行都根据K算出权重，然后又跟对应位置的V算出最终的值
# 所以是一一对应的
# ********************************************************************************

# 缩放点积注意力
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，num_queries，d)
    # keys的形状：(batch_size，num_keys，d)
    # values的形状：(batch_size，num_keys，value_size)
    # valid_lens的形状:(batch_size，)或者(batch_size，num_queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        # self.attention_weights的形状：(batch_size，num_queries，num_keys)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# ############################# 10.7 ##########################

# 定义注意力解码器
class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

# ############################# 10.7(old) ##########################
# def read_imdb(folder='train', data_root="/S1/CSCL/tangss/Datasets/aclImdb"):
#     data = []
#     for label in ['pos', 'neg']:
#         folder_name = os.path.join(data_root, folder, label)
#         for file in tqdm(os.listdir(folder_name)):
#             with open(os.path.join(folder_name, file), 'rb') as f:
#                 review = f.read().decode('utf-8').replace('\n', '').lower()
#                 data.append([review, 1 if label == 'pos' else 0])
#     random.shuffle(data)
#     return data
#
# def get_tokenized_imdb(data):
#     """
#     data: list of [string, label]
#     """
#     def tokenizer(text):
#         return [tok.lower() for tok in text.split(' ')]
#     return [tokenizer(review) for review, _ in data]
#
# def get_vocab_imdb(data):
#     tokenized_data = get_tokenized_imdb(data)
#     counter = collections.Counter([tk for st in tokenized_data for tk in st])
#     return torchtext.vocab.Vocab(counter, min_freq=5)
#
# def preprocess_imdb(data, vocab):
#     max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500
#
#     def pad(x):
#         return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))
#
#     tokenized_data = get_tokenized_imdb(data)
#     features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
#     labels = torch.tensor([score for _, score in data])
#     return features, labels
#
# def load_pretrained_embedding(words, pretrained_vocab):
#     """从预训练好的vocab中提取出words对应的词向量"""
#     embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为0
#     oov_count = 0 # out of vocabulary
#     for i, word in enumerate(words):
#         try:
#             idx = pretrained_vocab.stoi[word]
#             embed[i, :] = pretrained_vocab.vectors[idx]
#         except KeyError:
#             oov_count += 1
#     if oov_count > 0:
#         print("There are %d oov words." % oov_count)
#     return embed
#
# def predict_sentiment(net, vocab, sentence):
#     """sentence是词语的列表"""
#     device = list(net.parameters())[0].device
#     sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
#     label = torch.argmax(net(sentence.view((1, -1))), dim=1)
#     return 'positive' if label.item() == 1 else 'negative'

# ############################# 10.8 ##########################



# ############################# 10.9 ##########################

#@save
# 为了多注意力头的并行计算而变换形状
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 经过reshape后:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 经过permute后:(batch_size，num_heads，查询或者“键－值”对的个数,num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出:(batch_size*num_heads,查询或者“键－值”对的个数,num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

#@save
# 逆转transpose_qkv函数的操作
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

#@save
# 多头注意力
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    # 这里小改了一下教材，最终的输出可以指定，不一定是num_hiddens这么多（当然默认是）
    # 然后增加了个注意力权重的输出
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, num_outputs = None, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        if num_outputs is not None:
            self.W_o = nn.Linear(num_hiddens, num_outputs, bias=bias)
        else:
            self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，query_size or keys_size or values_size)
        # valid_lens的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        # ********************************************************************************
        # 这里这样干，主要是因为想要用一个全连接层直接训练多个头的Q\K\V
        # 所以这里输出的num_hiddens其实是hiddens_for_each_head*num_heads
        # 然后，算注意力的时候，每个batch之间不能互相干扰，多个头之间每个头也不能互相干扰
        # 所以直接把num_heads维塞入batch_sizes维
        # 将(batch_size,查询或者“键－值”对的个数,hiddens_for_each_head*num_heads)
        # 变成(batch_size*num_heads,查询或者“键－值”对的个数,hiddens_for_each_head)
        # 之后放入self.attention时，d2l.DotProductAttention计算中batch_size维是不会相互干扰的
        # torch.bmm只作用于同一batch的(num_queries，d)@(d,num_keys)，不会跨batch相乘
        # 也就是把heads放入batch维度，同样有不会跨heads相乘的效果
        # 最后逆过程把heads从batch维还原出来即可
        # 同时复制valid_lens以适应batch_size*num_heads的形状，在attention中mask
        # ********************************************************************************
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # 注意力权重
        self.attention_weights = transpose_output(self.attention.attention_weights, self.num_heads)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        # 最终输出：(batch_size，查询的个数，num_outputs)
        return self.W_o(output_concat)
