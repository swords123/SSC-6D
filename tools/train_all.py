import _init_paths

import argparse
import os
import numpy as np
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from lib.data.dataset_real import RealDataset
from lib.data.dataset_sym import SymDataset

from lib.models.network import PoseNet
from lib.loss.loss_sup import LossSup
#from lib.loss.loss_self import LossSelf
from lib.loss.loss_self_dist import LossSelfDIST

from itertools import cycle

from visdom import Visdom


parser = argparse.ArgumentParser(description='6D Pose Estimation')
parser.add_argument('--dataset_root', type=str, default = 'data/nocs')

parser.add_argument('--batch_size_sym', type=int, default = 12)
parser.add_argument('--batch_size_real', type=int, default = 2)

parser.add_argument('--workers_sym', type=int, default = 4)
parser.add_argument('--workers_real', type=int, default = 2)

parser.add_argument('--lr', default=0.0001)
parser.add_argument('--lr_rate', default=0.35)
parser.add_argument('--nepoch', type=int, default=12)
#parser.add_argument('--resume_posenet', type=str, default = 'laptop_pre.pth')
parser.add_argument('--resume_posenet', type=str, default = '')
parser.add_argument('--start_epoch', type=int, default = 1)

parser.add_argument('--start_real_epoch', type=int, default = 6)

parser.add_argument('--category', type=str, default='laptop')
parser.add_argument('--num_points', type=int, default=1024)
parser.add_argument('--model_points', type=int, default = 512)
parser.add_argument('--img_size', type=int, default = 192)
parser.add_argument('--render_size', type=int, default = 128)
parser.add_argument('--out_dir', type=str, default='output/')
parser.add_argument('--logger_freq', type=int, default = 10)
parser.add_argument('--deacy_epoch', type=int, default = 20)  

parser.add_argument("--deepsdf_dir", default="param/")

parser.add_argument("--model_name", default="")
parser.add_argument("--loss", type=str, default="a")
parser.add_argument('--aug', action="store_true", default=True)
                                                                           
opt = parser.parse_args()

opt.decay_epoch = [0,  3,  10,  13]
opt.decay_rate = [2.0, 1.0, 0.7, 0.35]

opt.out_dir = os.path.join(opt.out_dir, opt.category, opt.model_name)

vis = Visdom(env = "SCE7_" + opt.category + '_' + opt.model_name)

Beta_weight = {'bottle':3.0, 'bowl':2.5, 'can':2.5, 'laptop':1.5, 'mug':4.0, 'camera':3.0}


def train():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    print(opt)

    estimator = PoseNet(num_points = opt.num_points, num_obj = 1)
    estimator = nn.DataParallel(estimator)
    estimator.cuda()

    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    if opt.resume_posenet != '':
        # estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.out_dir, opt.resume_posenet)))
        estimator.load_state_dict(torch.load(opt.resume_posenet))

    loss_sup = LossSup(opt.model_points, opt.category)
    #loss_self = LossSelf(opt.img_size, opt.render_size)
    
    exp_dir = os.path.join(opt.deepsdf_dir, opt.category + '_remv')
    loss_self_dist = LossSelfDIST(opt.img_size, opt.render_size, exp_dir)

    dataset_sym = SymDataset('train', opt.category, opt.dataset_root, opt.num_points, opt.model_points, opt.img_size, opt.render_size)
    dataloader_sym = torch.utils.data.DataLoader(dataset_sym, batch_size=opt.batch_size_sym, shuffle=True, num_workers=opt.workers_sym, pin_memory=True)

    dataset_real = RealDataset('train', opt.category, opt.dataset_root, opt.num_points, opt.model_points, opt.img_size, opt.render_size, opt.aug)
    dataloader_real = torch.utils.data.DataLoader(dataset_real, batch_size=opt.batch_size_real, shuffle=True, num_workers=opt.workers_real, pin_memory=True)

    test_dataset_sym = SymDataset('test', opt.category, opt.dataset_root, opt.num_points, opt.model_points, opt.img_size, opt.render_size)
    testdataloader_sym = torch.utils.data.DataLoader(test_dataset_sym, batch_size=1, shuffle=False, num_workers=opt.workers_sym, pin_memory=True)

    test_dataset_real = RealDataset('test', opt.category, opt.dataset_root, opt.num_points, opt.model_points, opt.img_size, opt.render_size)
    testdataloader_real = torch.utils.data.DataLoader(test_dataset_real, batch_size=1, shuffle=False, num_workers=opt.workers_real, pin_memory=True)

    best_test_dis_sym = np.Inf
    best_test_code_sym = np.Inf
    best_test_dis_real = np.Inf

    # decay_start = False
    st_time = time.time()

    n_decays = len(opt.decay_epoch)
    assert len(opt.decay_rate) == n_decays
    for i in range(n_decays):
        if opt.start_epoch > opt.decay_epoch[i]:
            decay_count = i

    train_count = 0
    for epoch in range(opt.start_epoch, opt.nepoch+1):
        # if epoch >= opt.deacy_epoch and not decay_start:  
        #     decay_start = True
        #     opt.lr *= opt.lr_rate
        #     optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
        if decay_count < len(opt.decay_rate):
            if epoch > opt.decay_epoch[decay_count]:
                current_lr = opt.lr * opt.decay_rate[decay_count]
                optimizer = torch.optim.Adam(estimator.parameters(), lr=current_lr)
                decay_count += 1


        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>----train----<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        estimator.train()
        optimizer.zero_grad()

        dis_sym_sum = 0.0
        code_sym_sum = 0.0


        #loss_mask_from_pts_sum = 0.0

        loss_pts_from_pts_sum = 0.0

        loss_sdf_sum = 0.0
        #==========================================================
        dis_scale_sum = 0.0
        dis_trans_sum = 0.0
        code_norm_sum = 0.0
        #==========================================================

        time_st = time.time()

        for i, (data_sym, data_real) in enumerate(zip(dataloader_sym, cycle(dataloader_real))):
            points_sym = data_sym["points"].cuda()
            choose_sym = data_sym["choose"].cuda()
            img_sym = data_sym["rgb"].cuda()

            #target_sym = data_sym["target"].cuda()
            sRT_sym = data_sym['sRT'].cuda()
            model_sym = data_sym["model"].cuda()
            code_sym = data_sym["code"].cuda()

            pred_r_sym, pred_t_sym, pred_s_sym, pred_code_sym = estimator(img_sym, points_sym, choose_sym)
            # if epoch <= 1:
            #     _, dis_sym, code_loss_sym, _ = loss_sup(pred_r_sym, pred_t_sym, pred_s_sym, pred_code_sym, sRT_sym, model_sym, code_sym)
            # else:
            #     _, dis_sym, code_loss_sym, _ = loss_sup.forward_for_symm(pred_r_sym, pred_t_sym, pred_s_sym, pred_code_sym, sRT_sym, model_sym, code_sym)
            _, dis_sym, code_loss_sym, _ = loss_sup(pred_r_sym, pred_t_sym, pred_s_sym, pred_code_sym, sRT_sym, model_sym, code_sym)


            #loss = dis_sym * 0.75 + code_loss_sym
            loss = dis_sym + code_loss_sym
            
            dis_sym_sum += dis_sym.item()
            code_sym_sum += code_loss_sym.item()

            if epoch >= opt.start_real_epoch:
                points_real = data_real["points"].cuda()
                points_outliner_removed_real = data_real["points_outliner_removed"].cuda()
                depth_outliner_removed_gt_real = data_real['depth_outliner_removed_gt'].cuda()

                choose_real = data_real["choose"].cuda()
                img_real = data_real["rgb"].cuda()

                bbox = data_real['bbox'].cuda()
                intrix = data_real['intrix'].cuda()
                mask = data_real['mask'].cuda()

                pred_r_real, pred_t_real, pred_s_real, pred_code_real = estimator(img_real, points_real, choose_real)


                #====================================================================
                code_norm_value = torch.norm(pred_code_real, dim=1).mean()
                code_norm_sum = code_norm_sum + code_norm_value.item()

                loss_code_norm = torch.mean(torch.exp(torch.norm(pred_code_real * Beta_weight[opt.category], dim=1)) - 1.0) 
                #====================================================================

                loss_dict, to_show = loss_self_dist.forward(pred_r_real, pred_t_real, pred_s_real, pred_code_real, points_outliner_removed_real, depth_outliner_removed_gt_real, bbox, intrix, mask)


                # loss_dict = {
                #     "loss_gemo_mean": loss_gemo_mean,
                #     "loss_mask_gt_mean": loss_mask_gt_mean,
                #     "loss_mask_out_mean": loss_mask_out_mean,
                #     "loss_depth_mean": loss_depth_mean,
                #     "loss_pts_from_pts_mean": loss_pts_from_pts_mean,
                #     "loss_mask_from_pts_mean": loss_mask_from_pts_mean
                # }

                # to_show = {
                #     "mask_gt2s": mask_gt2s,
                #     "mask_re2s": mask_re2s,
                #     "false_mask_gt": false_mask_gt,
                #     "false_mask_out": false_mask_out,
                #     "pts_est": pts_est,
                #     "pts_gt": pts_gt,
                #     "pts_est_from_pt": pts_est_from_pt,
                #     "pts_gt_from_gt": pts_gt_from_gt,
                #     "pts_2d": pts_2d,
                #     "mask_2d": mask_2d
                # }

                #loss_mask_from_pts = loss_dict['loss_mask_from_pts_mean']

                loss_pts_from_pts = loss_dict['loss_pts_from_pts_mean']

                loss_sdf = loss_dict['loss_sdf_mean']               

                ###################################################################################################
                if opt.loss == 'a':  # the final loss we used !!!
                    loss = loss + loss_pts_from_pts * 100 + loss_sdf + loss_code_norm * 0.01
                elif opt.loss == 'b':      # ablation study
                    loss = loss + loss_sdf + loss_code_norm * 0.01
                elif opt.loss == 'c':      # remove syn loss
                    loss = loss + loss_pts_from_pts * 100 + loss_code_norm * 0.01
                elif opt.loss == 'd':      # remove syn loss
                    loss = loss + loss_pts_from_pts * 100 + loss_sdf                    
                elif opt.loss == 'real':  # remove syn loss
                    loss = loss_pts_from_pts * 100 + loss_sdf + loss_code_norm * 0.01
                else:
                    raise ValueError('opt.loss must be in [a, b, c, real]!')
                ###################################################################################################

                #loss_mask_from_pts_sum += loss_mask_from_pts.item()

                loss_pts_from_pts_sum += loss_pts_from_pts.item()

                loss_sdf_sum += loss_sdf.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_count += 1

            if train_count % opt.logger_freq == 0:
                print('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis_sym:{4} code_sym:{5} loss_sdf_sum:{6}'.format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), 
                    epoch, train_count, 
                    train_count * opt.batch_size_sym, 
                    dis_sym_sum / opt.logger_freq, 
                    code_sym_sum / opt.logger_freq, 
                    loss_sdf_sum / opt.logger_freq)
                )
                
                if train_count % (opt.logger_freq) == 0:
                        time_et = time.time()

                        vis.line(Y=np.array([time_et - time_st]), X=np.array([train_count]),
                                win=('train time'),
                                opts=dict(title='train time'),
                                update=None if train_count == 0 else 'append')

                        vis.line(Y=np.array([dis_sym_sum / opt.logger_freq]), X=np.array([train_count]),
                                win=('dis_sym_sum'),
                                opts=dict(title='dis_sym_sum'),
                                update=None if train_count == 0 else 'append')
                        vis.line(Y=np.array([code_sym_sum / opt.logger_freq]), X=np.array([train_count]),
                                win=('code_sym_sum'),
                                opts=dict(title='code_sym_sum'),
                                update=None if train_count == 0 else 'append')
                        
                        vis.line(Y=np.array([current_lr]), X=np.array([train_count]),
                                win=('current_lr'),
                                opts=dict(title='current_lr'),
                                update=None if train_count == 0 else 'append')


                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        vis.line(Y=np.array([loss_pts_from_pts_sum / opt.logger_freq]), X=np.array([train_count]),
                                    win=('loss_pts_from_pts_sum'),
                                    opts=dict(title='loss_pts_from_pts_sum'),
                                    update=None if train_count == 0 else 'append')

                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                        vis.line(Y=np.array([loss_sdf_sum / opt.logger_freq]), X=np.array([train_count]),
                                    win=('loss_sdf_sum'),
                                    opts=dict(title='loss_sdf_sum'),
                                    update=None if train_count == 0 else 'append')

                        #======================================================================================
                        vis.line(Y=np.array([code_norm_sum / opt.logger_freq]), X=np.array([train_count]),
                                    win=('code_norm_sum'),
                                    opts=dict(title='code_norm_sum'),
                                    update=None if train_count == 0 else 'append')
                        #======================================================================================


                        if epoch >= opt.start_real_epoch:
                            mask_gt = to_show['mask_gt2s'].to(torch.float)
                            mask_est = to_show['mask_re2s'].to(torch.float)
                            vis.image(mask_gt, win=('mask_gt'), opts=dict(title='mask_gt'))
                            vis.image(mask_est, win=('mask_est'), opts=dict(title='mask_est'))
                            
                            vis.scatter(X=torch.cat([to_show['pts_gt_from_gt'], to_show['pts_est_from_pt']], dim=0),
                                        Y=torch.cat([torch.ones(to_show['pts_gt_from_gt'].shape[0], dtype=torch.int32), 2 * torch.ones(to_show['pts_est_from_pt'].shape[0], dtype=torch.int32)], dim=0),
                                        win=('pts_all'),
                                        opts=dict(title='pts_all', markersize=2, markercolor=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.int32)))


                code_norm_sum = 0.0

                dis_sym_sum = 0.0
                code_sym_sum = 0.0

                loss_pts_from_pts_sum = 0.0

                loss_sdf_sum = 0.0

                time_st = time.time()

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>----test----<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        test_dis_sym = 0.0
        test_code_sym = 0.0
        test_dis_real = 0.0
        estimator.eval()

        test_count = 0
        for i, data_sym in enumerate(testdataloader_sym, 0):
            points_sym = data_sym["points"].cuda()
            choose_sym = data_sym["choose"].cuda()
            img_sym = data_sym["rgb"].cuda()

            #target_sym = data_sym["target"].cuda()
            sRT_sym = data_sym['sRT'].cuda()
            model_sym = data_sym["model"].cuda()
            code_sym = data_sym["code"].cuda()
            pred_r_sym, pred_t_sym, pred_s_sym, pred_code_sym = estimator(img_sym, points_sym, choose_sym)
            _, dis_sym, code_loss_sym, _ = loss_sup.forward_for_symm(pred_r_sym, pred_t_sym, pred_s_sym, pred_code_sym, sRT_sym, model_sym, code_sym)

            test_dis_sym += dis_sym.item()
            test_code_sym += code_loss_sym.item()
            test_count += 1
        
            if test_count % (opt.logger_freq * 10) == 0:
                print('Test time {0} Test Sym Frame No.{1} dis:{2} code:{3}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis_sym, code_loss_sym))
            
        test_dis_sym = test_dis_sym / test_count
        test_code_sym = test_code_sym / test_count


        test_count = 0
        for i, data_real in enumerate(testdataloader_real, 0):
            points_real = data_real["points"].cuda()
            choose_real = data_real["choose"].cuda()
            img_real = data_real["rgb"].cuda()

            #target_real = data_real["target_points"].cuda()
            sRT_real = data_real['sRT'].cuda()
            model_real = data_real["model_points"].cuda()
            pred_r_real, pred_t_real, pred_s_real, pred_code_real = estimator(img_real, points_real, choose_real)

            _, dis_real, _ = loss_sup.forward_gemo_for_symm(pred_r_real, pred_t_real, pred_s_real, sRT_real, model_real)

            test_dis_real += dis_real.item()
            test_count += 1
        
            if test_count % (opt.logger_freq * 10) == 0:
                print('Test time {0} Test Real Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis_real))
            
        test_dis_real = test_dis_real / test_count

        print('Test time {0} Epoch {1} TEST FINISH Avg Dis Sym: {2} Code Sym: {3} Avg Dis Real: {4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis_sym, test_code_sym, test_dis_real))
            
        vis.line(Y=np.array([test_dis_sym]), X=np.array([epoch]),
            win=('test_dis_sym'),
            opts=dict(title='test_dis_sym'),
            update=None if epoch == opt.start_epoch else 'append')
        vis.line(Y=np.array([test_code_sym]), X=np.array([epoch]),
            win=('test_code_sym'),
            opts=dict(title='test_code_sym'),
            update=None if epoch == opt.start_epoch else 'append')
        vis.line(Y=np.array([test_dis_real]), X=np.array([epoch]),
            win=('test_dis_real'),
            opts=dict(title='test_dis_real'),
            update=None if epoch == opt.start_epoch else 'append')

        # if test_code_sym <= best_test_code_sym:
        #     best_test_code_sym = test_code_sym
        #     torch.save(estimator.state_dict(), '{0}/pose_model_{1}_code_sym_{2}.pth'.format(opt.out_dir, epoch, test_code_sym))
        #     print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
        #     continue

        torch.save(estimator.state_dict(), '{0}/pose_model_{1}_dis_real_{2}.pth'.format(opt.out_dir, epoch, test_dis_real))

        #if test_dis_real <= best_test_dis_real:
        #    best_test_dis_real = test_dis_real
        #    torch.save(estimator.state_dict(), '{0}/pose_model_{1}_dis_real_{2}.pth'.format(opt.out_dir, epoch, test_dis_real))
        #    print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
        #    continue
    
        #if test_dis_sym <= best_test_dis_sym:
        #    best_test_dis_sym = test_dis_sym
        #    torch.save(estimator.state_dict(), '{0}/pose_model_{1}_dis_sym_{2}.pth'.format(opt.out_dir, epoch, test_dis_sym))
        #    print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
        #    continue


if __name__ == "__main__":
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    train()








