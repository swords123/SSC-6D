from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import time
import numpy as np
import torch.nn as nn
import random

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
        PerspectiveCameras, 
        RasterizationSettings, 
        MeshRasterizer, 
        MeshRenderer, 
        BlendParams,
        SoftSilhouetteShader
    )

from pytorch3d.loss import chamfer_distance

import matplotlib.pyplot as plt
# import open3d as o3d

class LossSelfPytorch3D(_Loss):
    def __init__(self, img_size, render_size):
        super(LossSelfPytorch3D, self).__init__(True)
        self.device = 'cuda'
        
        self.img_size = img_size
        self.render_size = render_size

        self.cameras = PerspectiveCameras(device=self.device)
        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        self.raster_settings = RasterizationSettings(
            image_size=self.render_size, 
            blur_radius=np.log(1. / 1e-4 - 1.) * self.blend_params.sigma, 
            faces_per_pixel=100, 
        )

        self.rasterizer=MeshRasterizer(
            cameras=self.cameras, 
            raster_settings=self.raster_settings
        )
        self.silhouette_renderer = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=SoftSilhouetteShader(blend_params=self.blend_params)
        )

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

    def render(self, mesh_v, mesh_e, R, t, bbox, intrix):
        rmin, rmax, cmin, cmax = bbox
        cam_fx, cam_fy, cam_cx, cam_cy = intrix

        img_size = (cmax - cmin, rmax - rmin)
        focal_length = (cam_fx, cam_fy)

        mesh = Meshes(verts = mesh_v, faces = mesh_e)
        mesh = mesh.cuda()

        rast_result = self.rasterizer(
            mesh, 
            R=R, T=t, principal_point=((cmax - cam_cx, rmax - cam_cy),),
            image_size=(img_size,),
            focal_length=(focal_length,)
        )

        silhouete = self.silhouette_renderer(
            mesh, 
            R=R, T=t, principal_point=((cmax - cam_cx, rmax - cam_cy),),
            image_size=(img_size,),
            focal_length=(focal_length,)
        )

        depth = rast_result.zbuf[0, :,:, 0]
        silhouete = silhouete[0, ...,3]

        depth = torch.flip(depth, [0,1])
        silhouete = torch.flip(silhouete, [0,1])

        return depth, silhouete


    def cal_chamfer_loss(self, depth, points, intrix, bbox):
        yy, xx = torch.meshgrid(torch.arange(depth.shape[0], device=self.device),
                                torch.arange(depth.shape[1], device=self.device))

        rmin, rmax, cmin, cmax = bbox
        cam_fx, cam_fy, cam_cx, cam_cy = intrix

        ratio_x = (cmax - cmin) / self.render_size
        ratio_y = (rmax - rmin) / self.render_size

        z = depth
        x = (xx * ratio_x + cmin - cam_cx) * z / cam_fx
        y = (yy * ratio_y + rmin - cam_cy) * z / cam_fy
 
        render_points = torch.stack([x.reshape(-1),y.reshape(-1),z.reshape(-1)], dim=1)
        render_points = render_points[torch.where(z.reshape(-1) > 0.00001)]

        chamfer_loss, _ = chamfer_distance(render_points.unsqueeze(0), points.unsqueeze(0))

        return chamfer_loss, (render_points, points)

    def cal_mask_loss(self, silhouete, mask):
        mask_loss = F.binary_cross_entropy(silhouete, mask, reduction='mean')
        #mask_loss = torch.norm((silhouete - mask))  * 0.001

        return mask_loss

    def forward(self, pred_r, pred_t, pred_s, points, mask, mesh, bbox, intrix):
        bs, _ = pred_r.size()
        R_martix = self.q2R(pred_r)
        s_martix = pred_s.contiguous().view(bs, 1, 1).repeat(1,3,3).view(bs, 3, 3)
        R_martix = s_martix * R_martix

        t_martix = pred_t.contiguous().view(bs, 3)

        loss_gemo_mean = 0.0
        loss_mask_mean = 0.0

        for i in range(bs):
            R_single = R_martix[i].unsqueeze(0).transpose(2, 1)
            t_single = t_martix[i].unsqueeze(0)
            mesh_v = mesh['v'][i][:mesh['v_len'][i]].unsqueeze(dim=0)
            mesh_e = mesh['e'][i][:mesh['e_len'][i]].unsqueeze(dim=0)
    
            depth, silhouete = self.render(mesh_v, mesh_e, R_single, t_single, bbox[i], intrix[i])

            mask_single = mask[i]

            c_loss, (pts_est, pts_gt) = self.cal_chamfer_loss(depth, points[i], intrix[i], bbox[i])
            m_loss = self.cal_mask_loss(silhouete, mask_single)

            loss_gemo_mean = loss_gemo_mean + c_loss
            loss_mask_mean = loss_mask_mean + m_loss

        loss_gemo_mean = loss_gemo_mean / bs
        loss_mask_mean = loss_mask_mean / bs

        return loss_gemo_mean, loss_mask_mean, (mask_single, silhouete, pts_est, pts_gt)# for vis
