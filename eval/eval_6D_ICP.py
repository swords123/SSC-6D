import argparse,copy,glob,json,os,time
from plyfile import PlyData
from tqdm import tqdm
from typing import OrderedDict
import _pickle as cPickle
import cv2
import numpy as np
import open3d as o3d
import torch
import torchvision.transforms as transforms

import sys
sys.path.append('./')
from lib.models.network import PoseNet
from lib.render.deep_sdf_decoder import Decoder
from lib.reconstruct.mesh import create_mesh
from eval.utils import compute_mAP, get_bbox, load_depth, q2R


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = 'data/nocs/')
parser.add_argument('--model_name', type=str)
parser.add_argument('--category', type=str, default='')
parser.add_argument('--out_dir', type=str, default='eval_results')
parser.add_argument('--estimator_model', type=str, default='', help='which model will be use')
parser.add_argument('--icp', action="store_true", default=False)
parser.add_argument('--num_points', type=int, default=1024, help='points number for point cloud')
parser.add_argument('--model_points', type=int, default = 512, help='points number for 3D model')
parser.add_argument('--img_size', type=int, default = 192)
parser.add_argument('--resolution', type=int, default=64)
opt = parser.parse_args()

icp_thres = {'bottle':0.005, 'bowl':0.001, 'camera':0.03, 'can':0.005, 'laptop':0.005, 'mug':0.005}
threshold = icp_thres[opt.category]
CLASS_MAP_FOR_CATEGORY = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}

opt.data='real_test'
file_path = 'Real/test_list.txt'
model_file_path = 'obj_models/real_test.pkl'
cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084

if not os.path.exists(os.path.join(opt.out_dir, opt.category, opt.model_name)):
    os.makedirs(os.path.join(opt.out_dir, opt.category, opt.model_name))

opt.out_dir = os.path.join(opt.out_dir, opt.category, opt.model_name)

opt.decoder_model = 'param/{}_remv/ModelParameters/2000.pth'.format(opt.category)
opt.specs_filename = 'param/{}_remv/specs.json'.format(opt.category)

xmap = np.array([[i for i in range(640)] for _ in range(480)])
ymap = np.array([[j for _ in range(640)] for j in range(480)])
norm_scale = 1000.0
norm_color = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

def project_vecs_onto_sphere(vectors, radius):
    new_vectors = torch.zeros_like(vectors)
    for i in range(vectors.shape[0]):
        v = vectors[i]
        length = torch.norm(v).detach()
        if length.cpu().data.numpy() > radius:
            new_vectors[i] = vectors[i].mul(radius / (length + 1e-8))
        else:
            new_vectors[i] = vectors[i]
    return new_vectors

def detect():
    print(opt)
    estimator = PoseNet(num_points=opt.num_points, num_obj=1)
    specs = json.load(open(opt.specs_filename))
    decoder = Decoder(latent_size=specs["CodeLength"], **specs["NetworkSpecs"])
    estimator.cuda()
    decoder.cuda()
    est_model = torch.load(opt.estimator_model)
    dec_model = torch.load(opt.decoder_model)["model_state_dict"]
    new_est_model = OrderedDict()
    new_dec_model = OrderedDict()
    for key in est_model:
        new_est_model[key.replace('module.', '')] = est_model[key]
    for key in dec_model:
        new_dec_model[key.replace('module.', '')] = dec_model[key]
    estimator.load_state_dict(new_est_model)
    estimator.eval()
    
    decoder.load_state_dict(new_dec_model)
    decoder.eval()

    # get test data list
    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
         for line in open(os.path.join(opt.dataset_root, file_path))]
    # frame by frame test
    t_inference = 0.0
    inst_count = 0
    img_count = 0

    if not os.path.exists(os.path.join(opt.out_dir, 'results_pkl')):
        os.makedirs(os.path.join(opt.out_dir, 'results_pkl'))
    if not os.path.exists(os.path.join(opt.out_dir, 'results_icp_pkl')) and opt.icp:
        os.makedirs(os.path.join(opt.out_dir, 'results_icp_pkl')) 
    if not os.path.exists(os.path.join(opt.out_dir, 'rec_mesh')):
        os.makedirs(os.path.join(opt.out_dir, 'rec_mesh'))

    t_start = time.time()

    for path in tqdm(img_list):
        img_path = os.path.join(opt.dataset_root, path)
        raw_rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        raw_rgb = raw_rgb[:, :, ::-1]
        raw_depth = load_depth(img_path)
        # load mask-rcnn detection results
        img_path_parsing = img_path.split('/')
        mrcnn_path = os.path.join(opt.dataset_root, 'results/mrcnn_results', opt.data, 'results_{}_{}_{}.pkl'.format(
            opt.data.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        with open(mrcnn_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)

        # find some category's index
        idx = np.argwhere(mrcnn_result['class_ids'] == CLASS_MAP_FOR_CATEGORY[opt.category]).squeeze(1)

        num_insts = idx.shape[0]
        f_sRT = np.zeros((num_insts, 4, 4), dtype=float)
        f_size = np.zeros((num_insts, 3), dtype=float)
        # prepare frame data
        valid_inst = []
        f_points, f_rgb, f_choose = [], [], []

        if opt.icp:
            f_points_icp = []

        for j, i in enumerate(idx):

            rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['rois'][i], raw_rgb.shape[0], raw_rgb.shape[1])
            mask = np.logical_and(mrcnn_result['masks'][:, :, i], raw_depth > 0)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            
            if len(choose) < 32:
                f_sRT[j] = np.identity(4, dtype=float)
                continue
            else:
                valid_inst.append(j)

            if opt.icp:
                depth_icp = raw_depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                xmap_icp = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                ymap_icp = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]

                pt2_icp = depth_icp / norm_scale
                pt2_icp = np.clip(pt2_icp, np.mean(pt2_icp) - 0.3, np.mean(pt2_icp) + 0.3)
                pt0_icp = (xmap_icp - cam_cx) * pt2_icp / cam_fx
                pt0_icp = np.clip(pt0_icp, np.mean(pt0_icp) - 0.3, np.mean(pt0_icp) + 0.3)
                pt1_icp = (ymap_icp - cam_cy) * pt2_icp / cam_fy
                pt1_icp = np.clip(pt1_icp, np.mean(pt1_icp) - 0.3, np.mean(pt1_icp) + 0.3)

                points_icp = np.concatenate((pt0_icp, pt1_icp, pt2_icp), axis=1)

            # process objects with valid depth observation
            if len(choose) > opt.num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:opt.num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, opt.num_points-len(choose)), 'wrap')
            depth_masked = raw_depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]

            pt2 = depth_masked / norm_scale
            pt2 = np.clip(pt2, np.mean(pt2) - 0.3, np.mean(pt2) + 0.3)
            pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
            pt0 = np.clip(pt0, np.mean(pt0) - 0.3, np.mean(pt0) + 0.3)
            pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
            pt1 = np.clip(pt1, np.mean(pt1) - 0.3, np.mean(pt1) + 0.3)

            points = np.concatenate((pt0, pt1, pt2), axis=1)
            rgb = raw_rgb[rmin:rmax, cmin:cmax, :]
            rgb = cv2.resize(rgb, (opt.img_size, opt.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = norm_color(rgb)

            crop_w = rmax - rmin
            ratio = opt.img_size / crop_w
            col_idx = choose % crop_w
            row_idx = choose // crop_w
            choose = (np.floor(row_idx * ratio) * opt.img_size + np.floor(col_idx * ratio)).astype(np.int64)
            # concatenate instances
            f_points.append(points)
            f_rgb.append(rgb)
            f_choose.append(choose)

            if opt.icp:
                f_points_icp.append(points_icp)

        if len(valid_inst):
            f_points = torch.cuda.FloatTensor(f_points)
            f_rgb = torch.stack(f_rgb, dim=0).cuda()
            f_choose = torch.cuda.LongTensor(f_choose)
            t_now = time.time()
            pred_r, pred_t, pred_s, pred_code = estimator(f_rgb, f_points, f_choose)
            pred_code = project_vecs_onto_sphere(pred_code, 1.0)
            t_inference += (time.time() - t_now)
            bs, _ = pred_r.size()
            r_matrix = q2R(pred_r)
            s = pred_s.contiguous().view(bs, 1, 1).repeat(1, 3, 3).view(bs, 3, 3)
            rs_matrix = r_matrix * s
            f_sRT[valid_inst, :3, :3] = rs_matrix.detach().cpu().numpy()
            f_sRT[valid_inst, :3, 3] = pred_t.detach().cpu().numpy()
            f_sRT[valid_inst, 3, 3] = np.ones((len(valid_inst),))

            image_short_path = '_'.join(img_path_parsing[-3:])

            for j, i in enumerate(valid_inst):
                mesh_path = os.path.join(opt.out_dir, 'rec_mesh','{0}_{1}'.format(image_short_path, j))
                if os.path.exists(mesh_path+'.ply'):
                    point_ele = PlyData.read(mesh_path+'.ply')['vertex']
                    mesh_points = np.stack([point_ele['x'],point_ele['y'],point_ele['z']],axis=1)
                else:
                    mesh_points = create_mesh(decoder, pred_code[j], N=opt.resolution,
                                            filename=mesh_path
                                            )
                f_size[i] = np.amax(np.abs(mesh_points), axis=0)
                
                if opt.icp:
                    part_pc = o3d.geometry.PointCloud()
                    part_pc.points = o3d.utility.Vector3dVector(f_points_icp[j])
                    mesh_pc = o3d.geometry.PointCloud()
                    mesh_pc.points = o3d.utility.Vector3dVector(mesh_points)

                    sRT = copy.deepcopy(f_sRT[i])
                    trans_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
                    trans_method.with_scaling = False
                    sRT[:3, :3] = sRT[:3, :3] * 0.5
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                            mesh_pc, part_pc, threshold, sRT,
                            trans_method,
                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
                    f_sRT[i] = reg_p2p.transformation
                    f_sRT[i][:3, :3] = f_sRT[i][:3, :3] * 2
                
        img_count += 1
        inst_count += len(valid_inst)

        # save results
        result = {}
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        result['gt_class_ids'] = gts['class_ids']
        result['gt_instance_ids'] = gts['instance_ids']
        result['gt_bboxes'] = gts['bboxes']
        result['gt_RTs'] = gts['poses']
        result['gt_scales'] = gts['size']
        result['gt_handle_visibility'] = gts['handle_visibility']

        result['pred_class_ids'] = mrcnn_result['class_ids'][idx]
        result['pred_bboxes'] = mrcnn_result['rois'][idx, :]
        result['pred_scores'] = mrcnn_result['scores'][idx]
        result['pred_RTs'] = f_sRT
        result['pred_scales'] = f_size

        image_short_path = '_'.join(img_path_parsing[-3:])
        if opt.icp:
            save_path = os.path.join(opt.out_dir, 'results_icp_pkl','results_{}.pkl'.format(image_short_path))
        else:
            save_path = os.path.join(opt.out_dir, 'results_pkl','results_{}.pkl'.format(image_short_path))
        with open(save_path, 'wb') as f:
            cPickle.dump(result, f)

    # write statistics
    fw = open('{0}/eval_logs.txt'.format(opt.out_dir), 'w')
    messages = ["=================================================================",
                "time:{0}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                "ICP: {0} thres: {1}".format(opt.icp, threshold),
                "Estimator model path: {}".format(opt.estimator_model),
                "Total images: {}".format(len(img_list)),
                "Valid images: {},  Total instances: {},  Average: {:.2f}/image".format(
                    img_count, inst_count, inst_count / img_count),
                "Inference time: {:06f}  Average: {:06f}/image".format(t_inference, t_inference / img_count),
                "Total time: {:06f}".format(time.time() - t_start)]
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()


def evaluate():
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    if opt.icp:
        result_pkl_list = glob.glob(os.path.join(opt.out_dir, 'results_icp_pkl', 'results_*.pkl'))
    else:
        result_pkl_list = glob.glob(os.path.join(opt.out_dir, 'results_pkl', 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list), 'no results'
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)

            if 'gt_handle_visibility' not in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False
    # To be consistent with NOCS, set use_matches_for_pose=True for mAP evaluation
    iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, opt.out_dir, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)
    # metric
    fw = open('{0}/eval_logs.txt'.format(opt.out_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    shift_10_idx = shift_thres_list.index(10)
    category_print = CLASS_MAP_FOR_CATEGORY[opt.category]

    messages = ['mAP:',
                '3D IoU at 25: {:.5f}'.format(iou_aps[category_print, iou_25_idx] * 100),
                '3D IoU at 50: {:.5f}'.format(iou_aps[category_print, iou_50_idx] * 100),
                '3D IoU at 75: {:.5f}'.format(iou_aps[category_print, iou_75_idx] * 100),
                '5 degree, 2cm: {:.5f}'.format(pose_aps[category_print, degree_05_idx, shift_02_idx] * 100),
                '5 degree, 5cm: {:.5f}'.format(pose_aps[category_print, degree_05_idx, shift_05_idx] * 100),
                '10 degree, 2cm: {:.5f}'.format(pose_aps[category_print, degree_10_idx, shift_02_idx] * 100),
                '10 degree, 5cm: {:.5f}'.format(pose_aps[category_print, degree_10_idx, shift_05_idx] * 100),
                '10 degree, 10cm: {:.5f}'.format(pose_aps[category_print, degree_10_idx, shift_10_idx] * 100),
                'Acc:',
                '3D IoU at 25: {:.5f}'.format(iou_acc[category_print, iou_25_idx] * 100),
                '3D IoU at 50: {:.5f}'.format(iou_acc[category_print, iou_50_idx] * 100),
                '3D IoU at 75: {:.5f}'.format(iou_acc[category_print, iou_75_idx] * 100),
                '5 degree, 2cm: {:.5f}'.format(pose_acc[category_print, degree_05_idx, shift_02_idx] * 100),
                '5 degree, 5cm: {:.5f}'.format(pose_acc[category_print, degree_05_idx, shift_05_idx] * 100),
                '10 degree, 2cm: {:.5f}'.format(pose_acc[category_print, degree_10_idx, shift_02_idx] * 100),
                '10 degree, 5cm: {:.5f}'.format(pose_acc[category_print, degree_10_idx, shift_05_idx] * 100),
                '10 degree, 10cm: {:.5f}'.format(pose_acc[category_print, degree_10_idx, shift_10_idx] * 100)
                ]
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()


if __name__ == '__main__':
    print('Detecting ...')
    detect()
    print('Detection Done!')
    print('Evaluating ...')
    evaluate()
    print('Evaluation Done!')
