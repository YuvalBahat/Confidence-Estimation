# From https://github.com/xternalz/WideResNet-pytorch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

CHECKPOINTS_PATH = 'STL10/'

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = int((depth - 4) / 6)
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        out = F.avg_pool2d(out, out.size()[2])
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out

def Load_Model(model,dataset,test_id,lr_4_training=None):
    num_classes = 100 if dataset=='cifar100' else 10
    if dataset == 'svhn':
        cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                         dropRate=0.4)
    elif dataset == 'stl10':
        cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,dropRate=0.3)
    else:
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)

    cnn = nn.DataParallel(cnn,device_ids=[0]).cuda()
    if lr_4_training is not None:
        cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=lr_4_training,momentum=0.9, nesterov=True, weight_decay=5e-4)
    model_path = CHECKPOINTS_PATH + test_id + '.pt'
    pretrained_exists = os.path.isfile(model_path)
    if pretrained_exists:
        state_dict = torch.load(model_path)
        latest_epoch = state_dict.pop('epoch',None)
        if 'model' in state_dict.keys():
            cnn.load_state_dict(state_dict['model'])
            if lr_4_training is not None:
                cnn_optimizer.load_state_dict(state_dict['optimizer'])
        else:#Supporting models saved before adding optimizer saving:
            cnn.load_state_dict(state_dict)
    else:
        latest_epoch = -1
    print('Loading a %s model for %s%s: %s'%(('pretrained ' if pretrained_exists else '')+model,dataset,', at epoch %d'%(latest_epoch) if pretrained_exists else '',model_path))
    if lr_4_training is None:
        return cnn,latest_epoch
    else:
        return cnn,latest_epoch,cnn_optimizer
