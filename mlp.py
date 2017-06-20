import torch
import torch.nn as nn
import random
import torchvision.transforms as transforms
import torch.utils.data as data
from dataset.imglist import NumData
from utils.trainer import Trainer
class MLP(nn.Module):
    def __init__(self, in_size = 2, hiddens = [4, 4, 4, 8], out_size = 3):
        super(MLP, self).__init__()
        layer_list = []
        in_ = in_size
        for item in hiddens:
            layer_list.append(nn.Linear(in_, item))
            layer_list.append(nn.ReLU(True))
            in_ = item
        layer_list.append(nn.Linear(in_, out_size))
        self.layers = nn.Sequential(*layer_list)
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()




import numpy as np
import matplotlib.pyplot as plt
means = [[0, 1], [1, 0], [1, 1]]
covs = [[[1, 0], [0, 10]], [[1, 0], [0, 0.1]], [[1, 0], [0, 1]]]

X = []
Y = []
for idx in range(3):
    x = np.random.multivariate_normal(means[idx], covs[idx], 1000)
    y = np.ones(1000)*idx
    X.append(x)
    Y.append(y)
X = np.concatenate(X, axis=0).astype('float32')
Y = np.concatenate(Y, axis=0).astype('int64')

train_loader = torch.utils.data.DataLoader(
    NumData(X, Y),
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True)

model = MLP()
model = torch.nn.DataParallel(model).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters())


model.train()
for epoch in range(10):
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var.view(input_var.size(0), -1))
        loss = criterion(output, target_var.view(input_var.size(0)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print loss.data.cpu().numpy()[0]