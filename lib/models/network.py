import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.models.pspnet import PSPNet

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}




class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()
        self.model = psp_models['resnet18'.lower()]()

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)

        self.conv1 = torch.nn.Conv1d(1408, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 512, 1)
        
        self.max_pool = torch.nn.MaxPool1d(num_points)

        self.fc1_r = torch.nn.Linear(512, 256)
        self.fc1_t = torch.nn.Linear(512, 256)
        self.fc1_s = torch.nn.Linear(512, 256)
        self.fc1_code = torch.nn.Linear(512, 256)

        self.fc2_r = torch.nn.Linear(256, 128)
        self.fc2_t = torch.nn.Linear(256, 128)
        self.fc2_s = torch.nn.Linear(256, 128)
        self.fc2_code = torch.nn.Linear(256, 128)

        self.fc3_r = torch.nn.Linear(128, 4)
        self.fc3_t = torch.nn.Linear(128, 3)
        self.fc3_s = torch.nn.Linear(128, 1)
        self.fc3_code = torch.nn.Linear(128, 16)

    def forward(self, img, x, choose):
        out_img = self.cnn(img)
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        #choose = choose.repeat(1, di, 1).view(bs, di, -1)
        choose = choose.view(bs, 1, -1).repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        x_mean = torch.mean(x, dim=1).view(bs, 1, -1)
        x = x - x_mean

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        ap_x_f = F.relu(self.conv1(ap_x))
        ap_x_f = F.relu(self.conv2(ap_x_f))

        mp_x = self.max_pool(ap_x_f).squeeze(2)

        rx = F.leaky_relu(self.fc1_r(mp_x))
        tx = F.leaky_relu(self.fc1_t(mp_x))
        sx = F.leaky_relu(self.fc1_s(mp_x))
        code_x = F.leaky_relu(self.fc1_code(mp_x))

        rx = F.leaky_relu(self.fc2_r(rx))
        tx = F.leaky_relu(self.fc2_t(tx))
        sx = F.leaky_relu(self.fc2_s(sx))
        code_x = F.leaky_relu(self.fc2_code(code_x))

        rx = self.fc3_r(rx)
        tx = self.fc3_t(tx)
        sx = torch.abs(self.fc3_s(sx))
        code_x = self.fc3_code(code_x)

        out_rx = rx.contiguous()
        out_tx = tx.contiguous()
        out_sx = sx.contiguous()
        out_code = code_x.contiguous()

        out_tx = out_tx + x_mean.view(bs, -1)
        
        return out_rx, out_tx, out_sx, out_code
