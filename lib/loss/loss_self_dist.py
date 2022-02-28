from matplotlib import colors
from numpy.lib.utils import byte_bounds
from torch.nn.modules import loss
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import time
import numpy as np
import torch.nn as nn
import random

from lib.render.renderer import SDFRenderer
from lib.render.utils.decoder_utils import load_decoder, decode_sdf

from pytorch3d.loss import chamfer_distance

import matplotlib.pyplot as plt
#import open3d as o3d

def construct_intrix_martix(intrix, bbox, render_size):
    cam_fx, cam_fy, cam_cx, cam_cy = intrix
    rmin, rmax, cmin, cmax = bbox

    ratio_x = render_size / (cmax - cmin)
    ratio_y = render_size / (rmax - rmin)

    cam_fx_new = cam_fx * ratio_x
    cam_fy_new = cam_fy * ratio_y
    cam_cx_new = (cam_cx - cmin) * ratio_x
    cam_cy_new = (cam_cy - rmin) * ratio_y

    intrix_martix = np.identity(3)
    intrix_martix[0,0] = cam_fx_new
    intrix_martix[1,1] = cam_fy_new
    intrix_martix[0,2] = cam_cx_new
    intrix_martix[1,2] = cam_cy_new
    return intrix_martix

def show_masks(mask_gt2s, mask_re2s, false_mask_gt, false_mask_out):
    mask_gt2s = mask_gt2s.detach().cpu().numpy()
    mask_re2s = mask_re2s.detach().cpu().numpy()
    false_mask_gt = false_mask_gt.detach().cpu().numpy()
    false_mask_out = false_mask_out.detach().cpu().numpy()
    
    fig = plt.figure()
    ax = fig.subplots(2,2)

    ax[0,0].imshow(mask_gt2s)
    ax[0,1].imshow(mask_re2s)
    ax[1,0].imshow(false_mask_gt)
    ax[1,1].imshow(false_mask_out) 

    plt.show()

def show_render_result(depth_rendered, mask_rendered, min_abs_query):
    depth_rendered = depth_rendered.detach().cpu().numpy()
    mask_rendered = mask_rendered.detach().cpu().numpy()
    min_abs_query = min_abs_query.detach().cpu().numpy()
    
    fig = plt.figure()
    ax = fig.subplots(2,2)

    ax[0,0].imshow(depth_rendered)
    ax[0,1].imshow(mask_rendered)
    ax[1,0].imshow(min_abs_query)

    plt.show()

class LossSelfDIST(_Loss):
    def __init__(self, img_size, render_size, exp_dir):
        super(LossSelfDIST, self).__init__(True)
        self.device = 'cuda'
        
        self.img_size = img_size
        self.render_size = render_size
        self.threshold = 5e-4

        self.transform_matrix = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        self.decoder = load_decoder(exp_dir, '2000', parallel=False)
        self.sdf_renderer = SDFRenderer(self.decoder, march_step=50, buffer_size=1, threshold=self.threshold, ray_marching_ratio=1.3)

    def project_vecs_onto_sphere(self, vectors, radius):
        new_vectors = torch.zeros_like(vectors)
        for i in range(vectors.shape[0]):
            v = vectors[i]
            length = torch.norm(v).detach()

            if length.cpu().data.numpy() > radius:
                new_vectors[i] = vectors[i].mul(radius / (length + 1e-8))
            else:
                new_vectors[i] = vectors[i]
        return new_vectors


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

    def depth2points(self, depth, bbox, intrix, mask = None):
        yy, xx = torch.meshgrid(torch.arange(depth.shape[0], device=self.device),
                                torch.arange(depth.shape[1], device=self.device))

        rmin, rmax, cmin, cmax = bbox
        cam_fx, cam_fy, cam_cx, cam_cy = intrix

        ratio_x = (cmax - cmin) / self.render_size
        ratio_y = (rmax - rmin) / self.render_size

        z = depth
        x = (xx * ratio_x + cmin - cam_cx) * z / cam_fx
        y = (yy * ratio_y + rmin - cam_cy) * z / cam_fy

        points = torch.stack([x.reshape(-1),y.reshape(-1),z.reshape(-1)], dim=1)

        if mask != None:
            valid_depth_mask = (depth > 0) & (depth < 1e5) # handle lidar data (sparse depth).
            valid_mask_overlap = mask & valid_depth_mask
            valid_mask_overlap = valid_mask_overlap.to(torch.bool)
            points = points[valid_mask_overlap.reshape(-1)]
        
        points = points[torch.where(points[:,2].reshape(-1) > 0.00001)]
        points = points[torch.where(points[:,2].reshape(-1) < 10000)]
        # #=======================
        # pcd_p = o3d.geometry.PointCloud()
        # pcd_p.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
        # o3d.visualization.draw_geometries([pcd_p])
        # #=======================

        return points


    def render(self, rotation, translation, scale, code, bbox, intrix):
        img_hw = (self.render_size, self.render_size)
        camera_intrix = construct_intrix_martix(intrix, bbox, self.render_size)
        camera_intrix = torch.from_numpy(camera_intrix).cuda()

        camera_extrix = torch.zeros((3,4)).cuda()
        camera_extrix[:,:3] = rotation
        camera_extrix[:,3] = translation / scale * 2.0

        self.sdf_renderer.set_intrinsic(camera_intrix, img_hw=img_hw, transform_matrix=self.transform_matrix)
        
        depth_rendered, mask_rendered, min_abs_query, points_min_sdf = self.sdf_renderer.render(code, camera_extrix[:,:3], camera_extrix[:,3], profile=False, sample_index_type='min_abs', ray_marching_type='pyramid_recursive')
        depth_rendered = depth_rendered * scale / 2.0

        #show_render_result(depth_rendered, mask_rendered, min_abs_query)

        return depth_rendered, mask_rendered, min_abs_query, points_min_sdf

    def compute_loss_points_from_depth(self, depth, points_gt, intrix, bbox):
        render_points = self.depth2points(depth, bbox, intrix)

        chamfer_loss, _ = chamfer_distance(render_points.unsqueeze(0), points_gt.unsqueeze(0))

        if torch.isnan(chamfer_loss):
            chamfer_loss = torch.tensor(0.0).cuda()
        
        return chamfer_loss, (render_points, points_gt)  

    def compute_loss_points_from_depth_v2(self, depth, mask, depth_gt, mask_gt, intrix, bbox):
        render_points = self.depth2points(depth, bbox, intrix, mask = mask)

        points_gt = self.depth2points(depth_gt, bbox, intrix, mask = mask_gt)

        chamfer_loss, _ = chamfer_distance(render_points.unsqueeze(0), points_gt.unsqueeze(0))

        if torch.isnan(chamfer_loss):
            chamfer_loss = torch.tensor(0.0).cuda()
        
        return chamfer_loss, (render_points, points_gt)    


    def compute_loss_points_from_points(self, points, points_gt, pred_r, pred_t, pred_s):
        pred_s = pred_s.contiguous().view(1, 1).repeat(3,3).view(3, 3)
        pred_r = pred_r * pred_s / 2.0
        pred_r = pred_r.contiguous().transpose(1, 0).contiguous()
        pred_t = pred_t.contiguous().view(1, 3)

        points_pred = torch.add(torch.mm(points, pred_r), pred_t)

        chamfer_loss, _ = chamfer_distance(points_pred.unsqueeze(0), points_gt.unsqueeze(0))

        if torch.isnan(chamfer_loss):
            chamfer_loss = torch.tensor(0.0).cuda()
        
        return chamfer_loss, (points_pred, points_gt) 

    def compute_loss_points_from_points_v2(self, points, depth_gt, mask_gt, pred_r, pred_t, pred_s, bbox, intrix):
        pred_s = pred_s.contiguous().view(1, 1).repeat(3,3).view(3, 3)
        pred_r = pred_r * pred_s / 2.0
        pred_r = pred_r.contiguous().transpose(1, 0).contiguous()
        pred_t = pred_t.contiguous().view(1, 3)

        points_pred = torch.add(torch.mm(points, pred_r), pred_t)

        points_gt = self.depth2points(depth_gt, bbox, intrix, mask=mask_gt)

        chamfer_loss, _ = chamfer_distance(points_pred.unsqueeze(0), points_gt.unsqueeze(0))

        if torch.isnan(chamfer_loss):
            chamfer_loss = torch.tensor(0.0).cuda()
        
        return chamfer_loss, (points_pred, points_gt) 


    #===========================================================================================

    # def get_min_sdf_sample_and_points(self, sdf_list, points_list, latent, index_type='min_abs', clamp_dist=0.1, profile=False, no_grad=False):
    #     #profiler = Profiler(silent = not profile)
    #     _, index = self.get_index_from_sdf_list(sdf_list, 1, index_type=index_type)
    #     points = self.collect_data_from_index(points_list, index)[0] # (N, 3)
    #     min_sdf_sample = decode_sdf(self.decoder, latent, points, clamp_dist=None, no_grad=no_grad).squeeze(-1)
    #     #profiler.report_process('[DEPTH] [SAMPLING] sample min sdf time\t')
    #     if no_grad:
    #         min_sdf_sample = min_sdf_sample.detach()
    #     return min_sdf_sample, points

    def compute_loss_sdf_align(self,points_pred, depth_gt, mask_gt, pred_r, pred_t, pred_s, pred_code, intrix, bbox):

        pred_s = pred_s.contiguous().view(1, 1).repeat(3,3).view(3, 3)
        pred_r = pred_r / pred_s * 2.0
        pred_r = pred_r.contiguous()#.transpose(1, 0).contiguous()
        pred_t = pred_t.contiguous().view(1, 3)

        #points_pred = torch.add(torch.mm(points, pred_r), pred_t)

        points_in_camera_cood = self.depth2points(depth_gt, bbox, intrix, mask=mask_gt)
        points_in_model_cood = torch.mm(torch.add(points_in_camera_cood, -pred_t), pred_r)

        points_distance_from_center = torch.norm(points_in_model_cood, dim=1)
        points_in_model_cood = points_in_model_cood[torch.where(points_distance_from_center < 0.98)]

        sdf = decode_sdf(self.decoder, pred_code, points_in_model_cood, clamp_dist=0.5).squeeze(-1)
        loss_sdf = torch.abs(sdf).mean()

        if torch.isnan(loss_sdf):
            loss_sdf = torch.tensor(0.0).cuda()

        return loss_sdf 
    #===========================================================================================


    def compute_loss_mask_from_points(self, points, mask_gt, pred_r, pred_t, pred_s, intrix, bbox):
        pred_s = pred_s.contiguous().view(1, 1).repeat(3,3).view(3, 3)
        pred_r = pred_r * pred_s / 2.0
        pred_r = pred_r.contiguous().transpose(1, 0).contiguous()
        pred_t = pred_t.contiguous().view(1, 3)

        points_pred = torch.add(torch.mm(points, pred_r), pred_t)

        rmin, rmax, cmin, cmax = bbox
        cam_fx, cam_fy, cam_cx, cam_cy = intrix

        ratio_x = (cmax - cmin) / self.render_size
        ratio_y = (rmax - rmin) / self.render_size

        xx = ((points_pred[:,0] * cam_fx) / points_pred[:, 2] + cam_cx - cmin) / ratio_x
        yy = ((points_pred[:,1] * cam_fy) / points_pred[:, 2] + cam_cy - rmin) / ratio_y

        points_2d = torch.stack([yy, xx], dim=1)
        mask_2d = torch.stack(torch.where(mask_gt), dim=1).to(torch.float32)

        # plt.figure()
        # plt.plot(points_2d[:, 0].detach().cpu().numpy(), points_2d[:, 1].detach().cpu().numpy(), 'bo', ms=5, color='red')
        # plt.plot(mask_2d[:, 0].detach().cpu().numpy(), mask_2d[:, 1].detach().cpu().numpy(), 'bo', ms=3, color='blue')
        # plt.show()

        chamfer_loss_2d, _ = chamfer_distance(points_2d.unsqueeze(0), mask_2d.unsqueeze(0))

        if torch.isnan(chamfer_loss_2d):
            chamfer_loss_2d = torch.tensor(0.0).cuda()
        
        return chamfer_loss_2d, (points_2d, mask_2d) 



    def compute_loss_depth(self, depth_output, valid_mask, depth_gt, valid_mask_gt):
        valid_depth_mask = (depth_gt > 0) & (depth_gt < 1e5) # handle lidar data (sparse depth).

        valid_mask_overlap = valid_mask & valid_mask_gt
        valid_mask_overlap = valid_mask_overlap & valid_depth_mask
        valid_mask_overlap = valid_mask_overlap.to(torch.bool)

        if torch.nonzero(valid_mask_overlap).shape[0] != 0:
            loss_depth = depth_output[valid_mask_overlap] - depth_gt[valid_mask_overlap]
        else:
            loss_depth = torch.zeros_like(valid_mask_overlap).float()

        loss_depth = torch.abs(loss_depth).mean()

        return loss_depth


    def compute_loss_mask(self, min_sdf_sample, valid_mask, valid_mask_gt, threshold=5e-5):
        # compute mask loss (gt \ out)
        false_mask_gt = valid_mask_gt & (~(valid_mask & valid_mask_gt))
        false_mask_gt = false_mask_gt.to(torch.bool)
        if torch.nonzero(false_mask_gt).shape[0] != 0:
            min_sdf_sample_gt = min_sdf_sample[false_mask_gt]
            loss_mask_gt = torch.max(min_sdf_sample_gt - threshold, torch.zeros_like(min_sdf_sample_gt))
        else:
            loss_mask_gt = torch.zeros_like(false_mask_gt).float()

        loss_mask_gt = loss_mask_gt.mean()

        false_mask_out = valid_mask & (~(valid_mask & valid_mask_gt))
        false_mask_out = false_mask_out.to(torch.bool)
        if torch.nonzero(false_mask_out).shape[0] != 0:
            min_sdf_sample_out = min_sdf_sample[false_mask_out]
            loss_mask_out = torch.max(-min_sdf_sample_out + threshold, torch.zeros_like(min_sdf_sample_out))
            #loss_mask_out = loss_mask_out.mean()
        else:
            loss_mask_out = torch.zeros_like(false_mask_out).float()
        loss_mask_out = loss_mask_out.mean()

        # plt.figure()
        # plt.imshow(false_mask_gt.detach().cpu().numpy())

        # plt.figure()
        # plt.imshow(false_mask_out.detach().cpu().numpy())
        return loss_mask_gt, loss_mask_out, (false_mask_gt, false_mask_out)


    #r_gt, t_gt, s_gt, pred_code_real, points_real, bbox, intrix, mask
    def forward(self, pred_r, pred_t, pred_s, pred_code, points, depth_gt, bbox, intrix, mask_gt):
        bs, _ = pred_r.size()
        pred_r_martix = self.q2R(pred_r)

        pred_code = self.project_vecs_onto_sphere(pred_code, 1.0)

        mask_gt = mask_gt.to(torch.uint8)

        #loss_gemo_mean = 0.0
        #loss_mask_gt_mean = 0.0
        #loss_mask_out_mean = 0.0

        #loss_depth_mean = 0.0

        loss_pts_from_pts_mean = 0.0
        #loss_mask_from_pts_mean = 0.0

        loss_sdf_mean = 0.0

        for i in range(bs):
            render_output = self.render(pred_r_martix[i], pred_t[i], pred_s[i], pred_code[i], bbox[i], intrix[i])
            depth_rendered, mask_rendered, min_abs_query, points_min_sdf = render_output

            points_min_sdf = points_min_sdf.detach()

            #loss_mask_gt, loss_mask_out, (false_mask_gt, false_mask_out) = self.compute_loss_mask(min_abs_query, mask_rendered, mask_gt[i], threshold = self.threshold)
            #c_loss, (pts_est, pts_gt) = self.compute_loss_points_from_depth(depth_rendered, points[i], intrix[i], bbox[i])
            #c_loss, (pts_est, pts_gt) = self.compute_loss_points_from_depth_v2(depth_rendered, mask_rendered, depth_gt[i], mask_gt[i], intrix[i], bbox[i])

            #loss_depth = self.compute_loss_depth(depth_rendered, mask_rendered, depth_gt[i], mask_gt[i])


            #loss_points_from_points, (pts_est_from_pt, pts_gt_from_gt)  = self.compute_loss_points_from_points(points_min_sdf, points[i], pred_r_martix[i], pred_t[i], pred_s[i])
            loss_points_from_points, (pts_est_from_pt, pts_gt_from_gt) = self.compute_loss_points_from_points_v2(points_min_sdf, depth_gt[i], mask_gt[i], pred_r_martix[i], pred_t[i], pred_s[i], bbox[i], intrix[i])

            #loss_mask_from_points, (pts_2d, mask_2d)  = self.compute_loss_mask_from_points(points_min_sdf, mask_gt[i], pred_r_martix[i], pred_t[i], pred_s[i], intrix[i], bbox[i])

            loss_sdf = self.compute_loss_sdf_align(points_min_sdf, depth_gt[i], mask_gt[i], pred_r_martix[i], pred_t[i], pred_s[i], pred_code[i], intrix[i], bbox[i])


            #loss_mask_gt_mean = loss_mask_gt_mean + loss_mask_gt
            #loss_mask_out_mean = loss_mask_out_mean + loss_mask_out

            #loss_gemo_mean = loss_gemo_mean + c_loss

            #loss_depth_mean = loss_depth_mean + loss_depth

            loss_pts_from_pts_mean = loss_pts_from_pts_mean + loss_points_from_points
            #loss_mask_from_pts_mean = loss_mask_from_pts_mean + loss_mask_from_points

            loss_sdf_mean = loss_sdf_mean + loss_sdf

            mask_gt2s = mask_gt[i]
            mask_re2s = mask_rendered
        
        #loss_gemo_mean = loss_gemo_mean / bs
        #loss_mask_gt_mean = loss_mask_gt_mean / bs
        #loss_mask_out_mean = loss_mask_out_mean / bs

        #loss_depth_mean = loss_depth_mean / bs

        loss_pts_from_pts_mean = loss_pts_from_pts_mean / bs
        #loss_mask_from_pts_mean = loss_mask_from_pts_mean / bs

        loss_sdf_mean = loss_sdf_mean / bs
        #show_masks(mask_gt2s, mask_re2s, false_mask_gt, false_mask_out)

        loss_dict = {
            #"loss_gemo_mean": loss_gemo_mean,
            #"loss_mask_gt_mean": loss_mask_gt_mean,
            #"loss_mask_out_mean": loss_mask_out_mean,
            #"loss_depth_mean": loss_depth_mean,
            "loss_pts_from_pts_mean": loss_pts_from_pts_mean,
            #"loss_mask_from_pts_mean": loss_mask_from_pts_mean,
            "loss_sdf_mean": loss_sdf_mean
        }

        to_show = {
            "mask_gt2s": mask_gt2s,
            "mask_re2s": mask_re2s,
            #"false_mask_gt": false_mask_gt,
            #"false_mask_out": false_mask_out,
            #"pts_est": pts_est,
            #"pts_gt": pts_gt,
            "pts_est_from_pt": pts_est_from_pt, 
            "pts_gt_from_gt": pts_gt_from_gt,
            #"pts_2d": pts_2d,
            #"mask_2d": mask_2d
        }


        return loss_dict, to_show
