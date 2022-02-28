import os, sys
import cv2
import math

import numpy as np
import _pickle as cPickle
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .utils import *

#import open3d as o3d
import matplotlib.pyplot as plt

CLASS_MAP_FOR_CATEGORY = {'bottle':1, 'bowl':2, 'camera':3, 'can':4, 'laptop':5, 'mug':6}

class SymDataset(data.Dataset):
    def __init__(self, mode, category, data_dir, n_pts, m_pts, img_size, mask_size):
        self.mode = mode
        self.category = CLASS_MAP_FOR_CATEGORY[category]
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.m_pts = m_pts
        self.img_size = img_size
        self.mask_size = mask_size

        assert mode in ['train', 'test']
        img_list_path = []
        model_file_path = []
        code_file_path = []

        if mode == 'train':
            # sampled data to max iters 50000 for one epoch
            #img_list_path = 'CAMERA/split/sym_train_' + category + '_remv_list.txt'
            img_list_path = 'param/'+ category + '_remv/split/sym_train_' + category + '_remv_sampled_list.txt'
            model_file_path = 'obj_models/camera_train.pkl'
            code_file_path = 'param/'+ category + '_remv/' + category + '_train_remv_lat.pkl'
        else:
            img_list_path = 'param/'+ category + '_remv/split/sym_val_' + category + '_list.txt'
            model_file_path = 'obj_models/camera_val.pkl' 
            code_file_path = 'param/'+ category + '_remv/' + category + '_val_remv_lat.pkl'

        self.img_list = [os.path.join('CAMERA', line.rstrip('\n')) for line in open(img_list_path)]
        self.length = len(self.img_list)

        models = {}
        with open(os.path.join(data_dir, model_file_path), 'rb') as f:
            models.update(cPickle.load(f))
        self.models = models

        codes = {}
        with open(code_file_path, 'rb') as f:
            codes.update(cPickle.load(f))
        self.codes = codes

        self.camera_intrinsics = [577.5, 577.5, 319.5, 239.5]
        self.norm_scale = 1000.0    
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.shift_range = 0.01
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        self.sym_list = [1, 2, 4]

        print('CAMERA dataset ' + self.mode + ': {} images found.'.format(self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_list[index])
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1]
        depth  = load_depth(img_path)
        mask_raw = cv2.imread(img_path + '_mask.png')[:, :, 2]

        cam_fx, cam_fy, cam_cx, cam_cy = self.camera_intrinsics

        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)

        idx = -1
        for i in range(len(list(gts['class_ids']))):
            if self.category == gts['class_ids'][i] and gts['model_list'][i] in self.codes:
                idx = i
                break
        if idx == -1:
            # for i in range(len(list(gts['class_ids']))):
            #     if self.category == gts['class_ids'][i]:
            #         print(self.category)
            #         print(gts['class_ids'][i])
            #         print(gts['model_list'][i])
            #         print(self.codes.keys())
            raise "Dataset Error!"

        #idx = list(gts['class_ids']).index(self.category)
        inst_id = gts['instance_ids'][idx]
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx], rgb.shape[0], rgb.shape[1])

        mask_raw = np.equal(mask_raw, inst_id)
        mask = np.logical_and(mask_raw, depth > 0)
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) >= self.n_pts:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.n_pts] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.n_pts-len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        pt2 = depth_masked / self.norm_scale
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        points = np.concatenate((pt0, pt1, pt2), axis=1)

        rgb = rgb[rmin:rmax, cmin:cmax, :]
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        crop_w = rmax - rmin
        ratio = self.img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)
        
        model = self.models[gts['model_list'][idx]].astype(np.float32)  
        sample_index = np.random.choice(model.shape[0], self.m_pts)
        model= model[sample_index, :]  

        code = self.codes[gts['model_list'][idx]].astype(np.float32) 

        scale = gts['scales'][idx]
        rotation = gts['rotations'][idx] 
        translation = gts['translations'][idx]

        if self.mode == 'train':
            rgb = self.colorjitter(Image.fromarray(np.uint8(rgb)))
            rgb = np.array(rgb)
            # add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
            # translation = translation + add_t[0]
            add_t = np.clip(0.001*np.random.randn(points.shape[0], 3), -0.005, 0.005)
            points = np.add(points, add_t)

        rgb = self.transform(rgb)
        points = points.astype(np.float32)

        if self.category in self.sym_list:
            rotation = gts['rotations'][idx]
            theta_x = rotation[0, 0] + rotation[2, 2]
            theta_y = rotation[0, 2] - rotation[2, 0]
            r_norm = math.sqrt(theta_x**2 + theta_y**2)
            s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                              [0.0,            1.0,  0.0           ],
                              [theta_y/r_norm, 0.0,  theta_x/r_norm]])
            rotation = rotation @ s_map

        sRT = np.identity(4, dtype=np.float32)
        sRT[:3, :3] = scale * rotation
        sRT[:3, 3] = translation
        #target = np.dot(model, sRT[:3, :3].T) + sRT[:3, 3]

        intrix = np.array([cam_fx, cam_fy, cam_cx, cam_cy], dtype=np.float32)

        mask_gt = mask_raw.astype(np.float32)
        mask_gt = cv2.resize(mask_gt[rmin:rmax, cmin:cmax], (self.mask_size, self.mask_size), interpolation=cv2.INTER_NEAREST) 

        sample = dict()
        sample['points'] = points
        sample['rgb'] = rgb
        sample['choose'] = choose
        
        sample['sRT'] = sRT
        #sample['target'] = target
        sample['model'] = model
        sample['code'] = code

        sample['scale'] = scale
        sample['rotation'] = rotation
        sample['translation'] = translation

        sample['bbox'] = np.array([rmin, rmax, cmin, cmax])
        sample['intrix'] = intrix
        sample['mask'] = mask_gt

        sample['model_name'] = gts['model_list'][idx]

        return sample

if __name__ == "__main__":
    dataset = SymDataset('train', 'laptop', 'data/nocs/', 1024, 512, 192)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=10)

    for i, data in enumerate(dataloader):
        points = data['points']
        rgb = data['rgb']
        choose = data['choose']
        target = data['target']
        model = data['model']
        code = data['code']
        points = points.numpy()[0]
        rgb = rgb.numpy()[0].transpose(1,2,0)

        plt.imshow(rgb)
        plt.show()

        color_red = np.array([[1.0,0.0,0.0]]*512)
        color_blue = np.array([[0.0,0.0,1.0]]*512)

        target = target.numpy()[0]

        break