import torch
from torch import nn

"""
    读取和存储
"""

# =======================
# 0. 选择设备
# =======================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =======================
# 1. 读写Tensor
# =======================

x = torch.ones(3, device=device)
torch.save(x, 'x.pt')
x2 = torch.load('x.pt')
print(x2)
y = torch.zeros(4, device=device)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)
torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
print(xy)

# =======================
# 2. 读写模型
# =======================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP().to(device)
# 输出所有可学习参数和缓冲区的名称及其对应的张量值
print(net.state_dict())

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 输出优化器的内部状态信息
print(optimizer.state_dict())

# 保存和加载模型
X = torch.randn(2, 3, device=device)
Y = net(X)

PATH = "./net.pt"
torch.save(net.state_dict(), PATH)

net2 = MLP().to(device)
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)

print(Y2 == Y)
"""
    tensor([[True],
            [True]], device='cuda:0')
"""

