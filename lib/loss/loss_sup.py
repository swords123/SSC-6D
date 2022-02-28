from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
# from lib.knn.__init__ import KNearestNeighbor

#import open3d as o3d

CLASS_MAP_FOR_CATEGORY = {'bottle':1, 'bowl':2, 'camera':3, 'can':4, 'laptop':5, 'mug':6}

class LossSup(_Loss):
    def __init__(self, num_points_mesh, category):
        super(LossSup, self).__init__(True)
        self.num_pt_mesh = num_points_mesh

        self.category = CLASS_MAP_FOR_CATEGORY[category]
        self.sym_list = [1, 2, 4]

        self.smoothl1 = nn.SmoothL1Loss()

    def q2R(self, pred_r):
        bs, _ = pred_r.size()
        pred_r = pred_r / (torch.norm(pred_r, dim=1).view(bs, 1))
        R_martix = torch.cat(((1.0 - 2.0*(pred_r[:, 2]**2 + pred_r[:, 3]**2)).view(bs, 1),\
                (2.0*pred_r[:, 1]*pred_r[:, 2] - 2.0*pred_r[:, 0]*pred_r[:, 3]).view(bs, 1), \
                (2.0*pred_r[:, 0]*pred_r[:, 2] + 2.0*pred_r[:, 1]*pred_r[:, 3]).view(bs, 1), \
                (2.0*pred_r[:, 1]*pred_r[:, 2] + 2.0*pred_r[:, 3]*pred_r[:, 0]).view(bs, 1), \
                (1.0 - 2.0*(pred_r[:, 1]**2 + pred_r[:, 3]**2)).view(bs, 1), \
                (-2.0*pred_r[:, 0]*pred_r[:, 1] + 2.0*pred_r[:, 2]*pred_r[:, 3]).view(bs, 1), \
                (-2.0*pred_r[:, 0]*pred_r[:, 2] + 2.0*pred_r[:, 1]*pred_r[:, 3]).view(bs, 1), \
                (2.0*pred_r[:, 0]*pred_r[:, 1] + 2.0*pred_r[:, 2]*pred_r[:, 3]).view(bs, 1), \
                (1.0 - 2.0*(pred_r[:, 1]**2 + pred_r[:, 2]**2)).view(bs, 1)), dim=1).contiguous().view(bs, 3, 3)
        return R_martix

    def cal_gemo_loss(self, pred_r, pred_t, pred_s, sRT, model_points):
        bs, _ = pred_r.size()
        base = self.q2R(pred_r)
        pred_s = pred_s.contiguous().view(bs, 1, 1).repeat(1,3,3).view(bs, 3, 3)
        base = base * pred_s

        base = base.contiguous().transpose(2, 1).contiguous()
        model_points = model_points.view(bs, self.num_pt_mesh, 3)

        sR_gt = sRT[:, :3, :3]
        T_gt = sRT[:, :3, 3].view(bs, 1, 3)
        target = torch.add(torch.bmm(model_points, sR_gt.transpose(2, 1)), T_gt)
        #target = target.view(bs, self.num_pt_mesh, 3)
        pred_t = pred_t.contiguous().view(bs, 1, 3)

        pred = torch.add(torch.bmm(model_points, base), pred_t)
        dis = torch.mean(torch.mean(torch.norm((pred - target), dim=2), dim=1))

        return pred, dis

    def norm_R(self, rotation):
        bs, _, _ = rotation.size()
        theta_x = rotation[:, 0, 0] + rotation[:, 2, 2]
        theta_y = rotation[:, 0, 2] - rotation[:, 2, 0]
        r_norm = torch.sqrt(theta_x * theta_x + theta_y * theta_y)

        s_map = torch.cat(((theta_x / r_norm).view(bs, 1),
                            torch.zeros_like(r_norm).view(bs,1).cuda(),
                            (-theta_y / r_norm).view(bs, 1),
                            torch.zeros_like(r_norm).view(bs,1).cuda(),
                            torch.ones_like(r_norm).view(bs,1).cuda(),
                            torch.zeros_like(r_norm).view(bs,1).cuda(),
                            (theta_y / r_norm).view(bs, 1),
                            torch.zeros_like(r_norm).view(bs,1).cuda(),
                            (theta_x / r_norm).view(bs, 1)), dim=1).contiguous().view(bs, 3, 3)

        rotation = torch.bmm(rotation, s_map)
        return rotation


    def cal_gemo_loss_sym(self, pred_r, pred_t, pred_s, sRT, model_points):
        bs, _ = pred_r.size()
        base = self.q2R(pred_r)
        #========================
        base = self.norm_R(base)
        #========================
        pred_s = pred_s.contiguous().view(bs, 1, 1).repeat(1,3,3).view(bs, 3, 3)
        base = base * pred_s

        base = base.contiguous().transpose(2, 1).contiguous()
        model_points = model_points.view(bs, self.num_pt_mesh, 3)

        sR_gt = sRT[:, :3, :3]
        T_gt = sRT[:, :3, 3].view(bs, 1, 3)
        target = torch.add(torch.bmm(model_points, sR_gt.transpose(2, 1)), T_gt)
        #target = target.view(bs, self.num_pt_mesh, 3)
        pred_t = pred_t.contiguous().view(bs, 1, 3)

        pred = torch.add(torch.bmm(model_points, base), pred_t)
        dis = torch.mean(torch.mean(torch.norm((pred - target), dim=2), dim=1))

        return pred, dis

    def cal_code_loss(self, pred_code, code):
        return torch.mean(torch.mean(torch.abs(pred_code - code), dim=1)) 
        #return self.smoothl1(pred_code, code)

    def forward(self, pred_r, pred_t, pred_s, pred_code, sRT, model_points, code):
        pred, dis_loss = self.cal_gemo_loss(pred_r, pred_t, pred_s, sRT, model_points)
        code_loss = self.cal_code_loss(pred_code, code)     
        loss = dis_loss + code_loss

        return loss, dis_loss, code_loss, pred

    def forward_gemo(self, pred_r, pred_t, pred_s, sRT, model_points):
        pred, dis_loss = self.cal_gemo_loss(pred_r, pred_t, pred_s, sRT, model_points)
        loss = dis_loss

        return loss, dis_loss, pred

    def forward_gemo_for_symm(self, pred_r, pred_t, pred_s, sRT, model_points):
        if self.category in self.sym_list:
            pred, dis_loss = self.cal_gemo_loss_sym(pred_r, pred_t, pred_s, sRT, model_points)
        else:
            pred, dis_loss = self.cal_gemo_loss(pred_r, pred_t, pred_s, sRT, model_points)
        loss = dis_loss

        return loss, dis_loss, pred
    
    def forward_for_symm(self, pred_r, pred_t, pred_s, pred_code, sRT, model_points, code):
        if self.category in self.sym_list:
            pred, dis_loss = self.cal_gemo_loss_sym(pred_r, pred_t, pred_s, sRT, model_points)
        else:
            pred, dis_loss = self.cal_gemo_loss(pred_r, pred_t, pred_s, sRT, model_points)
        code_loss = self.cal_code_loss(pred_code, code)     
        loss = dis_loss + code_loss

        return loss, dis_loss, code_loss, pred