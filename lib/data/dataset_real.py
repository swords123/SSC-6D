import os, sys, time
import cv2
import math
import copy
import random

import numpy as np
import _pickle as cPickle
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .utils import *

# from pytorch3d.structures import Meshes
# from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer, BlendParams
#import open3d as o3d
import matplotlib.pyplot as plt

CLASS_MAP_FOR_CATEGORY = {'bottle':1, 'bowl':2, 'camera':3, 'can':4, 'laptop':5, 'mug':6}

class RealDataset(data.Dataset):
    def __init__(self, mode, category, data_dir, n_pts, m_pts, img_size, mask_size, augment=True):
        self.mode = mode
        self.category = CLASS_MAP_FOR_CATEGORY[category]
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.m_pts = m_pts
        self.img_size = img_size
        self.mask_size = mask_size

        assert mode in ['train', 'test']

        if mode == 'train':
            if augment:
                img_list_path = 'param/'+ category + '_remv/split/real_train_' + category + '_sampled_list.txt'
            else:
                img_list_path = 'param/'+ category + '_remv/split/real_train_' + category + '_sampled_list_woAug.txt'
            model_file_path = 'obj_models/real_train/'
        else:
            img_list_path = 'param/'+ category + '_remv/split/real_test_' + category + '_sampled_list.txt'
            model_file_path = 'obj_models/real_test/'
                
        self.img_list = [os.path.join('Real', line.rstrip('\n')) for line in open(img_list_path)]
        self.length = len(self.img_list)

        models_mesh = {}
        models_points = {}

        suffixes = '_trans.obj' if category == 'mug' else '.obj'
        #suffixes = '.obj'
        for model_file in os.listdir(os.path.join(data_dir, model_file_path)):
            if category in model_file and suffixes in model_file:
                mesh = {}
                #model_id = model_file.split('.')[0]
                model_id = model_file.rstrip(suffixes)

                models_points[model_id], (v, e) = sample_points_from_mesh(os.path.join(data_dir, model_file_path,model_file), self.m_pts, fps=True, ratio=3)
                v_len = v.shape[0]
                e_len = e.shape[0]
                mesh['v'] = np.pad(v, ((0,50000 - v_len),(0,0)),'constant')
                mesh['e'] = np.pad(e, ((0,50000 - e_len),(0,0)),'constant')
                mesh['v_len'] = v_len
                mesh['e_len'] = e_len
                models_mesh[model_id] = mesh

        self.models_mesh = models_mesh

        self.models_points = models_points

        self.real_intrinsics_raw = [591.0125, 590.16775, 322.525, 244.11084]
        self.real_intrinsics_fliped = [591.0125, 590.16775, 317.475, 235.88916]


        self.norm_scale = 1000.0    # normalization scale
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.shift_range = 0.01
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        self.sym_list = [1, 2, 4]

        self.min_point_thresh = 100

        print('Real dataset ' + self.mode + ': {} images found.'.format(self.length))

    def __len__(self):
        return self.length

    def depth_to_points(self, choose, depth, bbox, real_intrix, is_clip = True):
        rmin, rmax, cmin, cmax = bbox
        cam_fx, cam_fy, cam_cx, cam_cy = real_intrix

        if len(choose) >= self.n_pts:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.n_pts] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.n_pts-len(choose)), 'wrap')

        depth_masked_img = depth[rmin:rmax, cmin:cmax]
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        
        pt2 = depth_masked / self.norm_scale
        if is_clip:
            pt2 = np.clip(pt2, np.mean(pt2) - 0.3, np.mean(pt2) + 0.3)

        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        if is_clip:
            pt0 = np.clip(pt0, np.mean(pt0) - 0.3, np.mean(pt0) + 0.3)

        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        if is_clip:
            pt1 = np.clip(pt1, np.mean(pt1) - 0.3, np.mean(pt1) + 0.3)
        
        points = np.concatenate((pt0, pt1, pt2), axis=1)

        return points, choose, depth_masked_img



    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_list[index])
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        
        rgb = rgb[:, :, ::-1]
        depth  = load_depth(img_path)

        #======================================================================
        depth_outliner_removed_path = img_path.replace('Real/', 'Real/depth/')
        depth_outliner_removed  = load_depth(depth_outliner_removed_path)
        #======================================================================

        mask_raw = cv2.imread(img_path + '_mask.png')[:, :, 2]

        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)


        if img_path.find('fliped') == -1:
            real_intrix = self.real_intrinsics_raw
        else:
            real_intrix = self.real_intrinsics_fliped
        cam_fx, cam_fy, cam_cx, cam_cy = real_intrix

        #=================================================================================
        idx = -1

        idx_random = list(range(len(list(gts['class_ids']))))
        random.shuffle(idx_random)
        
        for i in idx_random:
            if self.category == gts['class_ids'][i]:
                inst_id = gts['instance_ids'][i]
                rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][i], rgb.shape[0], rgb.shape[1])

                mask_raw_inst = np.equal(mask_raw, inst_id)
                mask_raw_points = np.logical_and(mask_raw_inst, depth > 0)
                choose = mask_raw_points[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

                mask_outliner_removed = np.logical_and(mask_raw_inst, depth_outliner_removed > 0)
                choose_outliner_removed = mask_outliner_removed[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

                if len(choose_outliner_removed) > self.min_point_thresh:
                    idx = i
                    break                
                # else:
                #     print('---------')
                #     print(i)
                #     print('---------')
                #     print(len(choose))
                #     print(len(choose_outliner_removed))
                #     print(img_path)
                #     print(gts)

        if idx == -1:
            raise "Dataset Error!"
        #=================================================================================
        if len(choose_outliner_removed) == 0:
            print('----------------------')

        points, choose, _ = self.depth_to_points(choose, depth, [rmin, rmax, cmin, cmax], real_intrix, is_clip = True)
        points_outliner_removed, choose_outliner_removed, depth_img_outliner_removed = self.depth_to_points(choose_outliner_removed, depth_outliner_removed, [rmin, rmax, cmin, cmax], real_intrix, is_clip=False)

        rgb = rgb[rmin:rmax, cmin:cmax, :]
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        crop_w = rmax - rmin
        ratio = self.img_size / crop_w

        #=================================================================================================
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)

        col_idx = choose_outliner_removed % crop_w
        row_idx = choose_outliner_removed // crop_w
        choose_outliner_removed = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)
        #================================================================================================
        
        scale = gts['scales'][idx]
        rotation = gts['rotations'][idx] 
        translation = gts['translations'][idx]

        if self.category in self.sym_list:
            rotation = gts['rotations'][idx]
            theta_x = rotation[0, 0] + rotation[2, 2]
            theta_y = rotation[0, 2] - rotation[2, 0]
            r_norm = math.sqrt(theta_x**2 + theta_y**2)
            s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                              [0.0,            1.0,  0.0           ],
                              [theta_y/r_norm, 0.0,  theta_x/r_norm]])
            rotation = rotation @ s_map

        model_mesh = {}
        model_mesh['v'] = copy.deepcopy(self.models_mesh[gts['model_list'][idx]]['v']).astype(np.float32)  / scale
        model_mesh['e'] = copy.deepcopy(self.models_mesh[gts['model_list'][idx]]['e'])
        model_mesh['v_len'] = self.models_mesh[gts['model_list'][idx]]['v_len']
        model_mesh['e_len'] = self.models_mesh[gts['model_list'][idx]]['e_len']

        model_points = self.models_points[gts['model_list'][idx]].astype(np.float32) / scale

        if self.mode == 'train':
            rgb = self.colorjitter(Image.fromarray(np.uint8(rgb)))
            rgb = np.array(rgb)
            add_t = np.clip(0.001*np.random.randn(points.shape[0], 3), -0.005, 0.005)

            points = np.add(points, add_t)
            points_outliner_removed = np.add(points_outliner_removed, add_t)
  
        rgb = self.transform(rgb)
        points = points.astype(np.float32)
        points_outliner_removed = points_outliner_removed.astype(np.float32)

        sRT = np.identity(4, dtype=np.float32)
        sRT[:3, :3] = scale * rotation
        sRT[:3, 3] = translation

        #target_points = np.dot(model_points, sRT[:3, :3].T) + sRT[:3, 3]

        intrix = np.array([cam_fx, cam_fy, cam_cx, cam_cy], dtype=np.float32)

        mask_gt = mask_raw_inst.astype(np.float32)
        mask_gt = cv2.resize(mask_gt[rmin:rmax, cmin:cmax], (self.mask_size, self.mask_size), interpolation=cv2.INTER_NEAREST) 
        depth_img_outliner_removed = depth_img_outliner_removed.astype(np.float32) / 1000.0
        depth_outliner_removed_gt = cv2.resize(depth_img_outliner_removed, (self.mask_size, self.mask_size), interpolation=cv2.INTER_NEAREST) 

        sample = dict()
        sample['points'] = points
        sample['rgb'] = rgb
        sample['choose'] = choose
        
        #supervised label
        sample['sRT'] = sRT
        #sample['target_points'] = target_points
        sample['model_points'] = model_points
        sample['scale'] = scale
        sample['rotation'] = rotation
        sample['translation'] = translation

        #unsupervised label
        sample['mesh'] = model_mesh
        sample['bbox'] = np.array([rmin, rmax, cmin, cmax])
        sample['intrix'] = intrix
        sample['mask'] = mask_gt

        sample['points_outliner_removed'] = points_outliner_removed
        sample['depth_outliner_removed_gt'] = depth_outliner_removed_gt

        sample['model_name'] = gts['model_list'][idx]
        # sample['file_name'] = [img_path]

        return sample

if __name__ == "__main__":
    dataset = RealDataset('train', 'mug', 'data/nocs/', 1024, 512, 192, 128)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=1)

    for i, data in enumerate(dataloader):
        points = data['points']
        sRT = data['sRT']

        # points_outliner_removed = data['points_outliner_removed']
        # rgb = data['rgb']
        # choose = data['choose']
        model = data['model_points']
        # mesh = data['mesh']
        # mask = data['mask']
        # bbox = data['bbox']
        # intrix = data['intrix']

        # depth = data['depth_outliner_removed_gt']
        # depth = depth.detach().cpu().numpy()[0]

        # plt.imshow(depth)
        # plt.show()

        sR_gt = sRT[:, :3, :3]
        T_gt = sRT[:, :3, 3].view(1, 1, 3)

        target = torch.add(torch.bmm(model, sR_gt.transpose(2, 1)), T_gt)

        points = points.detach().cpu().numpy()[0]
        pcd_p = o3d.geometry.PointCloud()
        pcd_p.points = o3d.utility.Vector3dVector(points)
        pcd_p.paint_uniform_color([1, 0, 0])

        target = target.detach().cpu().numpy()[0]
        pcd_t = o3d.geometry.PointCloud()
        pcd_t.points = o3d.utility.Vector3dVector(target)
        pcd_t.paint_uniform_color([0, 1, 0])

        o3d.visualization.draw_geometries([pcd_p, pcd_t])

        #break

        '''

        # mesh_v = mesh['v'][0][:mesh['v_len'][0]]
        # mesh_e = mesh['e'][0][:mesh['e_len'][0]]

        # print(mesh_v)

        # print(i)

        # points = points[0].numpy()

        # points_mean = np.mean(points, axis=0)
        # points_first = points[0]

        # deviation = np.sqrt(np.sum(np.square(points - points_mean), axis=1))

        # print(np.where(deviation > 0.4))

        # points[np.where(deviation > 0.4)] = points_first

        # pcd_p = o3d.geometry.PointCloud()
        # pcd_p.points = o3d.utility.Vector3dVector(points)

        # o3d.visualization.draw_geometries([pcd_p])


        scale = data['scale']
        rotation = data['rotation']
        scale = scale.view(8, 1, 1)
        rotation = rotation * scale

        print(scale.shape)
        print(rotation.shape)

        #rotation = scale * rotation

        translation = data['translation']

        cam_fx, cam_fy, cam_cx, cam_cy = intrix[0]
        rmin, rmax, cmin, cmax = bbox[0]

        mask = mask[0][rmin:rmax, cmin:cmax]

        mesh_v = mesh['v'][0][:mesh['v_len'][0]]
        mesh_e = mesh['e'][0][:mesh['e_len'][0]]
        # print(mesh_e.shape)
        # print(mesh_v.shape)
        
        img_size = (torch.tensor(cmax - cmin), torch.tensor(rmax - rmin))

        st = time.time()

        camera = PerspectiveCameras(focal_length=((cam_fx, cam_fy),), 
                                    #principal_point=((cmax - cam_cx, rmax - cam_cy),), 
                                    #R=rotation.transpose(2, 1)[0].unsqueeze(dim=0), T=translation[0].unsqueeze(dim=0), 
                                    #image_size=(img_size,),
                                    device='cuda')

        blend_params = BlendParams(sigma=1e-9, gamma=1e-9)
        raster_settings = RasterizationSettings(
            image_size=256, 
            blur_radius= np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=100, 
        )

        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        )

        mesh = Meshes(verts = mesh_v.unsqueeze(dim=0).to(torch.float32).cuda(), faces=mesh_e.unsqueeze(dim=0).cuda())

        rest_result = rasterizer(mesh, 
            R=rotation.transpose(2, 1)[0].unsqueeze(dim=0).cuda(), 
            T=translation[0].unsqueeze(dim=0).cuda(),
            principal_point=((cmax - cam_cx, rmax - cam_cy),),
            image_size=(img_size,)
            )
        
        et = time.time()
        print('time', et - st)

        depth = rest_result.zbuf[0, :,:, 0]
        depth = depth.cpu().numpy()
        depth = cv2.flip(depth,1)
        depth = cv2.flip(depth,0)

        depth = cv2.resize(depth, (cmax - cmin, rmax - rmin), interpolation=cv2.INTER_NEAREST)

        # xmap = np.array([[i for i in range(rmax - rmin)] for j in range(cmax - cmin)])
        # ymap = np.array([[j for i in range(rmax - rmin)] for j in range(cmax - cmin)])

        # z = depth.reshape(-1)
        # x = (xmap.flatten() - (cam_cx.numpy() - cmin.numpy())) * z / cam_fx.numpy()
        # y = (ymap.flatten() - (cam_cy.numpy() - rmin.numpy())) * z / cam_fx.numpy()

        # xyz=np.dstack((x,y,z))

        # np.save('render.npy', xyz)
        # np.save('point.npy', points[0].numpy())

        plt.figure()
        plt.imshow(depth + mask.numpy())
        #plt.imshow(depth)

        plt.figure()
        plt.imshow(mask)

        plt.show()

        # points = points.numpy()[0]
        # rgb = rgb.numpy()[0].transpose(1,2,0)
        # plt.imshow(rgb)
        # plt.show()
        # color_red = np.array([[1.0,0.0,0.0]]*512)
        # color_blue = np.array([[0.0,1.0,0.0]]*512)

        # target = target.numpy()[0]

        # pcd_p = o3d.geometry.PointCloud()
        # pcd_p.points = o3d.utility.Vector3dVector(points)
        # pcd_p.colors = o3d.utility.Vector3dVector(color_red)

        # pcd_t = o3d.geometry.PointCloud()
        # pcd_t.points = o3d.utility.Vector3dVector(target)
        # pcd_t.colors = o3d.utility.Vector3dVector(color_blue)

        # o3d.visualization.draw_geometries([pcd_p, pcd_t])

        #break
        '''