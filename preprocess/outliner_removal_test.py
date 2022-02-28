import os
import numpy as np
from PIL import Image
import cv2
import _pickle as cPickle

import numpngw

import open3d as o3d
import matplotlib.pyplot as plt

data_list_file = '/home/pwl/Work/Dataset/NOCS/nocs/Real/test_list_all.txt'
data_path = '/home/pwl/Work/Dataset/NOCS/nocs/Real/'
result_path = '/home/pwl/Work/Dataset/NOCS/nocs/Real/depth/'

cam_fx, cam_fy, cam_cx, cam_cy = [591.0125, 590.16775, 322.525, 244.11084]
xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])

def load_depth(img_path):
    """ Load depth image from img_path. """
    depth_path = img_path + '_depth.png'
    depth = cv2.imread(depth_path, -1)

    if len(depth.shape) == 3:
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def display_inlier_outlier_all(cloud, ind1, ind2, ind3):
    outlier_cloud_1 = cloud.select_by_index(ind1, invert=True)
    inlier_cloud_1 = cloud.select_by_index(ind1)

    outlier_cloud_2 = inlier_cloud_1.select_by_index(ind2, invert=True)
    inlier_cloud_2 = inlier_cloud_1.select_by_index(ind2)

    outlier_cloud_3 = inlier_cloud_2.select_by_index(ind3, invert=True)
    inlier_cloud_3 = inlier_cloud_2.select_by_index(ind3)

    outlier_cloud_1.paint_uniform_color([1, 0, 0])
    outlier_cloud_2.paint_uniform_color([0, 1, 0])
    outlier_cloud_3.paint_uniform_color([0, 0, 1])
    inlier_cloud_3.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud_3, outlier_cloud_1, outlier_cloud_2, outlier_cloud_3])

def depth2pc(depth, choose):
    depth_masked = depth.flatten()[choose][:, np.newaxis]
    xmap_masked = xmap.flatten()[choose][:, np.newaxis]
    ymap_masked = ymap.flatten()[choose][:, np.newaxis]
    pt2 = depth_masked
    pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
    points = np.concatenate((pt0, pt1, pt2), axis=1)

    return points


def outliner_removal(img_name):
    depth = load_depth(os.path.join(data_path, img_name))
    mask_raw = cv2.imread(os.path.join(data_path, img_name + '_mask.png'))[:, :, 2]
    gts = cPickle.load(open(os.path.join(data_path, img_name + '_label.pkl'), 'rb'))

    depth_flattened = depth.flatten()

    for idx in gts['instance_ids']:
        mask_i = np.equal(mask_raw, idx)
        mask_depth = np.logical_and(mask_i, depth > 0)
        #mask_depth = np.logical_and(mask_i, depth < 2500)
        choose = mask_depth.flatten().nonzero()[0]

        points_raw = depth2pc(depth, choose)

        #print(points_raw.shape)

        pcd_pc_raw = o3d.geometry.PointCloud()
        pcd_pc_raw.points = o3d.utility.Vector3dVector(points_raw)
        
        cl, ind_1 = pcd_pc_raw.remove_statistical_outlier(nb_neighbors=80, std_ratio=1.3)
        #display_inlier_outlier(pcd_pc_raw, ind_1)

        pcd_pc_inler1 = pcd_pc_raw.select_by_index(ind_1)
        cl, ind_2 = pcd_pc_inler1.remove_statistical_outlier(nb_neighbors=2000, std_ratio=4.5)
        #display_inlier_outlier(pcd_pc_inler1, ind_2)

        pcd_pc_inler2 = pcd_pc_inler1.select_by_index(ind_2)
        labels = np.array(pcd_pc_inler2.cluster_dbscan(eps=60, min_points=200))

        max_label = labels.max()
        min_label = labels.min()
        # print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd_pc_inler2.colors = o3d.utility.Vector3dVector(colors[:, :3])

        #o3d.visualization.draw_geometries([pcd_pc_inler2])

        biggest_cluster_idx = 0
        biggest_cluster_elem_count = 0
        if max_label >= 1 or min_label == -1:
            final_cluster_list = []
            for label_idx in range(max_label + 1):
                cluster_elem_count =  len(np.where(labels == label_idx)[0])
                if cluster_elem_count > biggest_cluster_elem_count:
                    biggest_cluster_elem_count = len(np.where(labels == label_idx)[0])
                    biggest_cluster_idx = label_idx
            
            final_cluster_list.append(biggest_cluster_idx)

            ind_biggest_cluster = np.where(labels == biggest_cluster_idx)[0]
            pcd_pc_biggest_cluster = pcd_pc_inler2.select_by_index(ind_biggest_cluster)
            pcd_pc_biggest_cluster_center = np.mean(np.array(pcd_pc_biggest_cluster.points), axis=0) 

            for label_idx in range(max_label + 1):
                if label_idx == biggest_cluster_idx:
                    continue
                label_idx_ind = np.where(labels == label_idx)[0]
                pcd_pc_idx_cluster = pcd_pc_inler2.select_by_index(label_idx_ind)

                pcd_pc_idx_cluster_center = np.mean(np.array(pcd_pc_idx_cluster.points), axis=0) 

                #print(np.linalg.norm(pcd_pc_biggest_cluster_center - pcd_pc_idx_cluster_center))
                if np.linalg.norm(pcd_pc_biggest_cluster_center - pcd_pc_idx_cluster_center) < 140:
                    final_cluster_list.append(label_idx)
            
            ind_3 = []
            for idx in final_cluster_list:
                idx_ind = list(np.where(labels == idx)[0])
                ind_3.extend(idx_ind)
            pcd_pc_inler = pcd_pc_inler2.select_by_index(ind_3)
        else:
            pcd_pc_inler = pcd_pc_inler2
            ind_3 = np.array(range(labels.shape[0]))

        #display_inlier_outlier_all(pcd_pc_raw, ind_1, ind_2, ind_3)
        #o3d.visualization.draw_geometries([pcd_pc_inler])

        choose_f1 = choose[ind_1]
        choose_del1 = np.delete(choose, ind_1)

        choose_f2 = choose_f1[ind_2]
        choose_del2 = np.delete(choose_f1, ind_2)
    
        choose_f3 = choose_f2[ind_3]
        choose_del3 = np.delete(choose_f2, ind_3)
        
        choose_deleted = np.concatenate([choose_del1, choose_del2, choose_del3])
        choose_final = choose_f3

        depth_flattened[choose_deleted] = 0

    depth_final = depth_flattened.reshape((480, 640))

    if not os.path.exists(os.path.join(result_path, img_name.split('/')[0], img_name.split('/')[1])):
        os.makedirs(os.path.join(result_path, img_name.split('/')[0], img_name.split('/')[1]))

    saved_path = os.path.join(result_path, img_name + '_depth.png')
    numpngw.write_png(saved_path, depth_final)

if __name__ == "__main__":
    data_list = open(data_list_file).readlines()
    data_list = [item.strip('\n') for item in  data_list]
    
    for img_name in data_list:
        image_name_array = img_name.split('/')

        print(img_name)
        outliner_removal(img_name)
        #break
