import numpy as np
import cv2
import os

import numpngw
import shutil

import matplotlib.pyplot as plt

raw_data_path = '/home/pwl/Work/Dataset/NOCS/nocs/Real/train'
depth_data_path = '/home/pwl/Work/Dataset/NOCS/nocs/Real/depth/train'

def load_depth(img_path):
    depth = cv2.imread(img_path, -1)

    if len(depth.shape) == 3:
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def flip_raw_data():
    scene_id_list = os.listdir(raw_data_path)
    for scene_id in scene_id_list:
        scene_dir = os.path.join(raw_data_path, scene_id)
        for file_name in os.listdir(scene_dir):
            img_path = os.path.join(scene_dir, file_name)
            print(img_path)
            if img_path.find('fliped_') != -1:
                continue
            
            if file_name.endswith('color.png') or file_name.endswith('coord.png') or file_name.endswith('mask.png'):
                img = cv2.imread(img_path)
                img_filped = cv2.flip(img, 1)
                
                ends = img_path.split('_')[-1]
                img_fliped_path = img_path.replace(ends, 'fliped_' + ends)
                cv2.imwrite(img_fliped_path, img_filped)

            if file_name.endswith('depth.png'):
                img = load_depth(img_path)
                img_filped = cv2.flip(img, 1)

                ends = img_path.split('_')[-1]
                img_fliped_path = img_path.replace(ends, 'fliped_' + ends)

                numpngw.write_png(img_fliped_path, img_filped)
            
            if file_name.endswith('meta.txt'):
                ends = img_path.split('_')[-1]
                img_fliped_path = img_path.replace(ends, 'fliped_' + ends)
                shutil.copy(img_path, img_fliped_path)


def flip_depth_data():
    scene_id_list = os.listdir(depth_data_path)
    for scene_id in scene_id_list:
        scene_dir = os.path.join(depth_data_path, scene_id)
        for file_name in os.listdir(scene_dir):
            img_path = os.path.join(scene_dir, file_name)
            print(img_path)
            if img_path.find('fliped_') != -1:
                continue

            if file_name.endswith('depth.png'):
                img = load_depth(img_path)
                img_filped = cv2.flip(img, 1)

                ends = img_path.split('_')[-1]
                img_fliped_path = img_path.replace(ends, 'fliped_' + ends)

                numpngw.write_png(img_fliped_path, img_filped)


if __name__ == "__main__":
    print('--------------flip_raw_data-----------------------')
    flip_raw_data()
    
    print('--------------flip_depth_data-----------------------')
    flip_depth_data()



