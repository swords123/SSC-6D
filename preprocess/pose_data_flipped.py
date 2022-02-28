import os
import sys
import glob
import cv2
import numpy as np
import _pickle as cPickle
from tqdm import tqdm
from align import align_nocs_to_depth
from utils import load_depth


def process_data(img_path, depth):
    """ Load instance masks for the objects in the image. """
    mask_path = img_path + '_mask.png'
    mask = cv2.imread(mask_path)[:, :, 2]
    mask = np.array(mask, dtype=np.int32)
    all_inst_ids = sorted(list(np.unique(mask)))
    assert all_inst_ids[-1] == 255
    del all_inst_ids[-1]    # remove background
    num_all_inst = len(all_inst_ids)
    h, w = mask.shape

    coord_path = img_path + '_coord.png'
    coord_map = cv2.imread(coord_path)[:, :, :3]
    coord_map = coord_map[:, :, (2, 1, 0)]
    # flip z axis of coord map
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    class_ids = []
    instance_ids = []
    model_list = []
    masks = np.zeros([h, w, num_all_inst], dtype=np.uint8)
    coords = np.zeros((h, w, num_all_inst, 3), dtype=np.float32)
    bboxes = np.zeros((num_all_inst, 4), dtype=np.int32)

    meta_path = img_path + '_meta.txt'
    with open(meta_path, 'r') as f:
        i = 0
        for line in f:
            line_info = line.strip().split(' ')
            inst_id = int(line_info[0])
            cls_id = int(line_info[1])
            # background objects and non-existing objects
            if cls_id == 0 or (inst_id not in all_inst_ids):
                continue
            if len(line_info) == 3:
                model_id = line_info[2]    # Real scanned objs
            else:
                model_id = line_info[3]    # CAMERA objs
            # remove one mug instance in CAMERA train due to improper model
            if model_id == 'b9be7cfe653740eb7633a2dd89cec754':
                continue
            # process foreground objects
            inst_mask = np.equal(mask, inst_id)
            # bounding box
            horizontal_indicies = np.where(np.any(inst_mask, axis=0))[0]
            vertical_indicies = np.where(np.any(inst_mask, axis=1))[0]
            assert horizontal_indicies.shape[0], print(img_path)
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            # object occupies full image, rendering error, happens in CAMERA dataset
            if np.any(np.logical_or((x2-x1) > 600, (y2-y1) > 440)):
                return None, None, None, None, None, None
            # not enough valid depth observation
            final_mask = np.logical_and(inst_mask, depth > 0)
            if np.sum(final_mask) < 64:
                continue
            class_ids.append(cls_id)
            instance_ids.append(inst_id)
            model_list.append(model_id)
            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))
            bboxes[i] = np.array([y1, x1, y2, x2])
            i += 1
    # no valid foreground objects
    if i == 0:
        return None, None, None, None, None, None

    masks = masks[:, :, :i]
    coords = np.clip(coords[:, :, :i, :], 0, 1)
    bboxes = bboxes[:i, :]

    return masks, coords, class_ids, instance_ids, model_list, bboxes


def annotate_real_train(data_dir):
    """ Generate gt labels for Real train data through PnP. """
    real_train = open(os.path.join(data_dir, 'Real/train_list_all_fliped.txt')).read().splitlines()
    intrinsics_raw = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    intrinsics_fliped = np.array([[591.0125, 0, 317.475], [0, 590.16775, 235.88916], [0, 0, 1]])
    # scale factors for all instances
    scale_factors = {}
    path_to_size = glob.glob(os.path.join(data_dir, 'obj_models/real_train', '*_norm.txt'))
    for inst_path in sorted(path_to_size):
        instance = os.path.basename(inst_path).split('.')[0]
        bbox_dims = np.loadtxt(inst_path)
        scale_factors[instance] = np.linalg.norm(bbox_dims)
    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    valid_img_list = []
    for img_path in tqdm(real_train):
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        
        #=================================================
        if img_full_path.find('fliped') == -1:
            #print(img_full_path)
            intrinsics = intrinsics_raw
        else:
            intrinsics = intrinsics_fliped
        #=================================================

        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue
        depth = load_depth(img_full_path)
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        if instance_ids is None:
            continue
        # compute pose
        ###########################################################################################################################
        #The object pose of the flipped image calculated by this code is wrong, but it does not affect our self-supervised training
        ###########################################################################################################################
        num_insts = len(class_ids)
        scales = np.zeros(num_insts)
        rotations = np.zeros((num_insts, 3, 3))
        translations = np.zeros((num_insts, 3))
        for i in range(num_insts):
            s = scale_factors[model_list[i]]
            mask = masks[:, :, i]
            idxs = np.where(mask)
            coord = coords[:, :, i, :]
            coord_pts = s * (coord[idxs[0], idxs[1], :] - 0.5)
            coord_pts = coord_pts[:, :, None]
            img_pts = np.array([idxs[1], idxs[0]]).transpose()
            img_pts = img_pts[:, :, None].astype(float)
            distCoeffs = np.zeros((4, 1))    # no distoration
            retval, rvec, tvec = cv2.solvePnP(coord_pts, img_pts, intrinsics, distCoeffs)
            assert retval
            R, _ = cv2.Rodrigues(rvec)
            T = np.squeeze(tvec)
            # re-label for mug category
            if class_ids[i] == 6:
                T0 = mug_meta[model_list[i]][0]
                s0 = mug_meta[model_list[i]][1]
                T = T - s * R @ T0
                s = s / s0
            scales[i] = s
            rotations[i] = R
            translations[i] = T
        # write results
        gts = {}
        gts['class_ids'] = class_ids    # int list, 1 to 6
        gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
        gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
        gts['rotations'] = rotations.astype(np.float32)    # np.array, R
        gts['translations'] = translations.astype(np.float32)  # np.array, T
        gts['instance_ids'] = instance_ids  # int list, start from 1
        gts['model_list'] = model_list  # str list, model id/name
        with open(img_full_path + '_label.pkl', 'wb') as f:
            cPickle.dump(gts, f)
        valid_img_list.append(img_path)
    # write valid img list to file
    with open(os.path.join(data_dir, 'Real/train_list_fliped.txt'), 'w') as f:
        for img_path in valid_img_list:
            f.write("%s\n" % img_path)


if __name__ == '__main__':
    data_dir = '/home/pwl/Work/Dataset/NOCS/nocs/'

    train_list_all = open(os.path.join(data_dir, 'Real/train_list_all.txt'))
    train_list_all_flipped = open(os.path.join(data_dir, 'Real/train_list_all_fliped.txt'), 'w')

    for line in train_list_all:
        line = line.strip('\n')
        line_fliped = line.strip('\n') + '_fliped'
        train_list_all_flipped.write(line + '\n')
        train_list_all_flipped.write(line_fliped + '\n')

    train_list_all.close()
    train_list_all_flipped.close()

    # create list for all data
    annotate_real_train(data_dir)
