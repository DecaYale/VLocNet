import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate
# from torchvision import transforms, models
from .ResNet import resnet_elu
from utils import Logger
import numpy as np
import os.path as osp
import time
import os
import copy
from utils.quaternion_math import quaternion_angular_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import pickle
import json
from collections import deque


def load_state_dict(model, state_dict):
    """
    Loads a state dict when the model has some prefix before the parameter names
    :param model:
    :param state_dict:
    :return: loaded model
    """
    model_names = [n for n, _ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

    # find prefix for the model and state dicts from the first param name
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        print('Could not find the correct prefixes between {:s} and {:s}'.
              format(model_names[0], state_names[0]))
        raise KeyError

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, '')
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


def safe_collate(batch):
    """
    Collate function for DataLoader that filters out None's
    :param batch: minibatch
    :return: minibatch filtered for None's
    """
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class Evaluator(object):
    experiment = 'VLocNet'

    def __init__(self, model, train_criterion, test_dataset, config, resume_optim=False):
        self.model = model
        self.train_criterion = train_criterion

        self.config = config
        self.logdir = config.logdir
        if(not os.path.isdir(self.logdir)):
            os.makedirs(self.logdir)
        # set random seed
        torch.manual_seed(self.config.seed)
        if self.config.cuda:
            torch.cuda.manual_seed(self.config.seed)

        self.start_epoch = int(0)
        assert(osp.isfile(self.config.checkpoint_file))

        loc_func = None if self.config.cuda else lambda storage, loc: storage
        checkpoint = torch.load(
            self.config.checkpoint_file, map_location=loc_func)

        load_state_dict(self.model, checkpoint['model_state_dict'])
        print('Loaded checkpoint {:s} epoch {:d}'.format(self.config.checkpoint_file,
                                                         checkpoint['epoch']))

        # dataloder
        self.val_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=self.config.batch_size, shuffle=False,
                                                      num_workers=self.config.num_workers, pin_memory=True,
                                                      collate_fn=safe_collate)

        # activate GPUs
        if self.config.cuda:
            self.model.cuda()
            self.train_criterion.cuda()
            # self.val_criterion.cuda()

        self.xq_global_p = None
        self.xq_gt_p = None
        self.pose_queue = deque()
        self.pose_p = None
        # loss functions
        self.t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
        self.q_criterion = quaternion_angular_error

        self.err_filename = osp.join(osp.expanduser(
            config.logdir), '{:s}.json'.format("Errs"))
        self.image_filename = osp.join(osp.expanduser(
            config.logdir), '{:s}.png'.format("figure"))
        self.result_filename = osp.join(osp.expanduser(
            config.logdir), '{:s}.pkl'.format("result"))

    def step_feedfwd(self, data, model, cuda, target=None, criterion=None, optim=None,
                     train=True, max_grad_norm=0.0):
        '''
        training/validation step for a feedforward NN
        :param data:
        :param target:
        :param model:
        :param criterion:
        :param optim:
        :param cuda: whether CUDA is to be used
        :param train: training / val stage
        :param max_grad_norm: if > 0, clips the gradient norm
        :return:
        '''
        if train:
            assert criterion is not None

        data_var = Variable(data, volatile=(not train),
                            requires_grad=train)  # ??requires_grad=train
        if cuda:
            data_var = data_var.cuda(async=True)

        if cuda:
            target = target.cuda(async=True)

        target_var = Variable(target, volatile=(
            not train), requires_grad=False)

        # images = data_var[:, 1:, ...].clone()
        # pose_p = target_var[:, :-1, ...].clone()
        images = data_var.clone()
        pose_p = target_var[:, :-1, ...].clone()
        # dummy pose_p for the oldest frame
        pose_p = torch.cat([torch.zeros_like(pose_p[:, 0:1]), pose_p], dim=1)

        if(self.pose_p is not None):
            pose_p = self.pose_p

        # expand the xq_global and target_var to include the previous frame
        # xq_global_T = torch.cat([torch.unsqueeze(self.xq_global_p, 1),
        #                          torch.unsqueeze(xq_global, 1)], dim=1)
        # target_var_T = torch.cat([torch.unsqueeze(self.xq_gt_p, 1),
        #                          torch.unsqueeze(target_var, 1)], dim=1)

        # loss = criterion(xq_odom, xq_global_T, target_var)

        # assert(len(self.pose_ps) == 2)
        # pose_p = torch.cat(self.pose_ps)

        # output = model(data_var)
        xq_odom, xq_global = model(
            (images, pose_p)
        )

        # update
        self.pose_p = xq_global.detach()

        if criterion is not None:
            # if cuda:
            #     target = target.cuda(async=True)

            # target_var = Variable(target, volatile=(
            #     not train), requires_grad=False)

            ####

            # loss = criterion(xq_odom, xq_global, target_var)
            # pose_loss, odom_loss = criterion(
            #     xq_odom, xq_global, target_var[:, 1:, ...])
            pose_loss, odom_loss = criterion(
                xq_odom, xq_global, target_var)
            loss = pose_loss + odom_loss

            return loss.data[0], (xq_odom, xq_global)
        else:
            return 0, (xq_odom, xq_global)

    def test(self):
        """
        Function that does the training and validation
        :param lstm: whether the model is an LSTM
        :return:
        """

        # for epoch in range(self.start_epoch, self.config.n_epochs):
        # eval
        # self.model.train()
        self.model.eval()
        pred_poses = []  # element: Nx7
        targ_poses = []  # element: Nx7

        for batch_idx, (data, target) in enumerate(self.val_loader):

            kwargs = dict(target=target, criterion=self.train_criterion,
                          optim=None, train=False)

            loss, pred = self.step_feedfwd(data, self.model, self.config.cuda,
                                           **kwargs)
            xq_global = pred[1]

            # print('xq_global.shape:', xq_global.size())
            size = xq_global.size()
            xq_global = xq_global[:, -
                                  1, ...].cpu().data.numpy().reshape((-1, size[-1]))
            pred_poses.append(xq_global)
            target = target[:, -
                            1, ...].cpu().data.numpy().reshape((-1, size[-1]))
            targ_poses.append(target)
            # train_batch_time.update(time.time() - end)
            # print(xq_global.shape, target.shape)
            end = time.time()

        # normalize quaternions
        print('pred_poses[0].shape:', pred_poses[0].shape)
        qs = [q[:, 3:]/(np.linalg.norm(q[:, 3:], axis=-1, keepdims=True)+1e-12)
              for q in pred_poses]
        # print(qs[0].shape)
        pred_poses = [np.hstack((pred_poses[i][:, :3], q))
                      for i, q in enumerate(qs)]
        # print(pred_poses[0].shape)
        pred_poses = np.vstack(pred_poses)

        # only select the current poses
        # targ_poses = [t[1] for t in targ_poses]
        targ_poses = np.vstack(targ_poses)
        fig = self.visualize(pred_poses, targ_poses)
        self.export(pred_poses, targ_poses, fig)
        # Save final checkpoint

        # print('Epoch {:d} checkpoint saved'.format(epoch))
        # if self.config['log_visdom']:
        #     self.vis.save(envs=[self.vis_env])

    def visualize(self, pred_poses, targ_poses):
        # create figure object
        fig = plt.figure()
        if self.config.dataset != '7Scenes':
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

        # plot on the figure object
        # ss = max(1, int(len(dataset) / 1000))  # 100 for stairs
        ss = 1
        # scatter the points and draw connecting line
        x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
        y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))

        if self.config.dataset != '7Scenes':  # 2D drawing
            ax.plot(x, y, c='b')
            ax.scatter(x[0, :], y[0, :], c='r')
            ax.scatter(x[1, :], y[1, :], c='g')
        else:

            z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
            for xx, yy, zz in zip(x.T, y.T, z.T):
                ax.plot(xx, yy, zs=zz, c='b', linewidth=0.2)
            ax.scatter(x[0, :], y[0, :], zs=z[0, :],
                       c='r', depthshade=0, s=0.8)
            ax.scatter(x[1, :], y[1, :], zs=z[1, :],
                       c='g', depthshade=0, s=0.8)
            ax.view_init(azim=119, elev=13)

        return fig

    def export(self, pred_poses, targ_poses, fig):

        t_loss = np.asarray([self.t_criterion(p, t) for p, t in zip(pred_poses[:, :3],
                                                                    targ_poses[:, :3])])
        q_loss = np.asarray([self.q_criterion(p, t) for p, t in zip(pred_poses[:, 3:],
                                                                    targ_poses[:, 3:])])

        errs = {
            "Error in translation(median)": "{:5.3f}".format(np.median(t_loss)),
            "Error in translation(mean)": "{:5.3f}".format(np.mean(t_loss)),
            "Error in rotation(median)": "{:5.3f}".format(np.median(q_loss)),
            "Error in rotation(mean)": "{:5.3f}".format(np.mean(q_loss)),
        }
        print(errs)
        with open(self.err_filename, 'w') as out:
            json.dump(errs, out)
        print('{:s} saved'.format(self.err_filename))

        fig.savefig(self.image_filename)
        print('{:s} saved'.format(self.image_filename))
        with open(self.result_filename, 'wb') as f:
            pickle.dump({'targ_poses': targ_poses,
                         'pred_poses': pred_poses}, f)
        print('{:s} saved'.format(self.result_filename))
