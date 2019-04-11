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
from tensorboardX import SummaryWriter
from utils.tensorboardx_summary import summarize
from utils.quaternion_math import quaternion_angular_error
from collections import OrderedDict
from .optimizer import Optimizer


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
    return model


def adapt_load_state_dict(model, state_dict):
    """
    Loads a state dict when the model has some prefix before the parameter names
    :param model:
    :param state_dict:
    :return: loaded model
    """
    model_names = [n for n, _ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

    # find prefix for the model and state dicts from the first param name
    # if model_names[0].find(state_names[0]) >= 0:
    #     model_prefix = model_names[0].replace(state_names[0], '')
    #     state_prefix = None
    # elif state_names[0].find(model_names[0]) >= 0:
    #     state_prefix = state_names[0].replace(model_names[0], '')
    #     model_prefix = None
    # else:
    #     print('Could not find the correct prefixes between {:s} and {:s}'.
    #           format(model_names[0], state_names[0]))
    #     raise KeyError

    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    import copy
    new_state_dict = copy.deepcopy(model.state_dict())
    shape_conflicts = []
    missed = []

    for k, v in new_state_dict.items():
        if k in state_dict:
            if v.size() == state_dict[k].size():
                new_state_dict[k] = state_dict[k]
            else:
                shape_conflicts.append(k)
        else:
            missed.append(k)

    if(len(missed) > 0):
        print("Warning: The flowing parameters are missed in checkpoint: ")
        print(missed)
    if (len(shape_conflicts) > 0):
        print(
            "Warning: The flowing parameters are fail to be initialized due to the shape conflicts: ")
        print(shape_conflicts)

    model.load_state_dict(new_state_dict)
    return model


def safe_collate(batch):
    """
    Collate function for DataLoader that filters out None's
    :param batch: minibatch
    :return: minibatch filtered for None's
    """
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class Trainer(object):
    experiment = 'VLocNet'

    def __init__(self, model, optimizer_config, train_criterion, train_dataset, val_dataset, config, resume_optim=False, val_criterion=None):
        self.model = model

        self.train_criterion = train_criterion
        if val_criterion is None:
            self.val_criterion = self.train_criterion
        else:
            self.val_criterion = val_criterion
        self.sep_train = optimizer_config['sep_train']
        optimizer_config.pop('sep_train', 0)
        self.n_epochs = config.n_epochs
        lr_decay = config.lr_decay
        lr_stepvalues = [k/4*self.n_epochs for k in range(1, 5)]
        print()
        if(self.sep_train):
            odom_param_list = [
                # self.train_criterion.sx_abs, self.train_criterion.sq_abs,
                # self.train_criterion.sx_rel, self.train_criterion.sq_rel,
                self.train_criterion.sx_vo, self.train_criterion.sq_vo
            ]
            odom_param_list.extend(
                [v for n, v in self.model.named_parameters() if 'odom' in n])

            pose_param_list = [
                self.train_criterion.sx_abs, self.train_criterion.sq_abs,
                self.train_criterion.sx_rel, self.train_criterion.sq_rel,
                # self.train_criterion.sx_vo, self.train_criterion.sq_vo
            ]
            # pose_param_list.extend(
            #     [v for n, v in self.model.named_parameters() if 'global' in n])
            pose_param_list.extend(
                [v for n, v in self.model.named_parameters() if 'global' in n or 'share' in n or 'odom_en2_head' in n])
            # print([n for n, v in self.model.named_parameters()
            #        if 'global' in n or 'share' in n or 'odom_en2_head' in n], '!!')

            # for n, v in self.model.named_parameters():
            #     if n not in [n for n, v in self.model.named_parameters() if 'odom' in n] + [n for n, v in self.model.named_parameters() if 'global' in n or 'share' in n or 'odom_en2_head' in n]:
            #         print(n)
            assert(len(odom_param_list) + len(pose_param_list)
                   - len([v for n, v in self.model.named_parameters()
                          if 'share' in n or 'odom_en2_head' in n])
                   == len(list(self.model.named_parameters()))+6
                   )
            self.odom_optimizer = Optimizer(
                params=odom_param_list, lr_decay=lr_decay, lr_stepvalues=lr_stepvalues, **optimizer_config)
            self.pose_optimizer = Optimizer(
                params=pose_param_list, lr_decay=lr_decay, lr_stepvalues=lr_stepvalues, **optimizer_config)
        else:
            self.optimizer = Optimizer(
                params=self.model.parameters(), lr_decay=self.lr_decay, lr_stepvalues=lr_stepvalues, **optimizer_config)

        self.config = config
        self.logdir = config.logdir
        if(not os.path.isdir(self.logdir)):
            os.makedirs(self.logdir)
        # set random seed
        torch.manual_seed(self.config.seed)
        if self.config.GPUs > 0:
            torch.cuda.manual_seed(self.config.seed)

        self.start_epoch = int(0)
        if self.config.checkpoint_file:
            if osp.isfile(self.config.checkpoint_file):
                loc_func = None if self.config.GPUs > 0 else lambda storage, loc: storage
                checkpoint = torch.load(
                    self.config.checkpoint_file, map_location=loc_func)

                # load model
                self.model = adapt_load_state_dict(self.model, checkpoint.get(
                    'model_state_dict', checkpoint))
                # print(self.model.state_dict()['odom_en1_head.conv1.weight'])
                # load_state_dict(self.model, checkpoint)
                if resume_optim:
                    self.optimizer.learner.load_state_dict(
                        checkpoint['optim_state_dict'])
                    self.start_epoch = checkpoint['epoch']
                    if checkpoint.has_key('criterion_state_dict'):
                        c_state = checkpoint['criterion_state_dict']
                        append_dict = {k: torch.Tensor([0.0])
                                       for k, _ in self.train_criterion.named_parameters()
                                       if not k in c_state}
                        c_state.update(append_dict)
                        self.train_criterion.load_state_dict(c_state)
                print('Loaded checkpoint {:s} epoch {:d}'.format(self.config.checkpoint_file,
                                                                 checkpoint.get('epoch', -1)))

        # dataloder
        self.recur_train = config.recur_train
        if self.recur_train:
            assert(self.config.skip == 1)
            shuffle = False
        else:
            shuffle = self.config.shuffle
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=self.config.batch_size, shuffle=shuffle,
                                                        num_workers=self.config.num_workers, pin_memory=True,
                                                        collate_fn=safe_collate)
        if self.config.do_val:
            self.val_loader = torch.utils.data.DataLoader(val_dataset,
                                                          batch_size=self.config.batch_size, shuffle=shuffle,
                                                          num_workers=self.config.num_workers, pin_memory=True,
                                                          collate_fn=safe_collate)
        else:
            self.val_loader = None

        self.model = nn.DataParallel(
            self.model, device_ids=range(self.config.GPUs))
        # activate GPUs
        if self.config.GPUs > 0:
            self.model.cuda()
            self.train_criterion.cuda()
            self.val_criterion.cuda()

        self.xq_global_p = None
        self.xq_gt_p = None
        self.t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
        self.q_criterion = quaternion_angular_error
        self.t_error_best = np.inf
        self.q_error_best = np.inf
        self.pose_p = None

        self.summary_writer = SummaryWriter(
            log_dir=osp.join(self.logdir, 'runs'))
        # summarize the graph
        input_size = 256 if self.config.crop_size < 0 else self.config.crop_size
        dummy_input = [
            Variable(torch.rand(1, 3, 3, input_size, input_size)),
            Variable(torch.rand(1, 3, 7))
        ]
        if self.config.GPUs > 0:
            dummy_input = [d.cuda() for d in dummy_input]
        if self.config.GPUs < 2:
            self.summary_writer.add_graph(self.model, dummy_input)
            pass
        # print(self.model.state_dict().keys())

    def save_checkpoint(self, epoch):

        print(self.sep_train, not self.sep_train)
        if not self.sep_train:
            optim_state_dict = self.optimizer.learner.state_dict()
        else:
            optim_state_dict = copy.deepcopy(
                self.odom_optimizer.learner.state_dict())
            optim_state_dict.update(
                self.pose_optimizer.learner.state_dict())

        filename = osp.join(self.logdir, 'epoch_{:03d}.pth.tar'.format(epoch))
        checkpoint_dict = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                           'optim_state_dict': optim_state_dict,
                           'criterion_state_dict': self.train_criterion.state_dict()}
        torch.save(checkpoint_dict, filename)

    def step_feedfwd(self, data, model, cuda, target=None, criterion=None, optim=None,
                     train=True, max_grad_norm=0.0, task='both'):
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
        :task: ['both', 'odom', 'pose']
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

        # output = model(data_var)
        # images = data_var[:, 1:, ...].clone()

        images = data_var.clone()
        pose_p = target_var[:, :-1, ...].clone()
        # dummy pose_p for the oldest frame
        pose_p = torch.cat([torch.zeros_like(pose_p[:, 0:1]), pose_p], dim=1)

        # when evaluating or recur_train flag is set
        if (not train or self.recur_train) and self.pose_p is not None:
            # pose_p = self.pose_p
            pose_p = torch.cat(
                [torch.zeros_like(self.pose_p[:, 0:1]), self.pose_p], dim=1)

        xq_odom, xq_global = model(
            (images, pose_p)
        )

        self.pose_p = xq_global.detach()

        if criterion is not None:
            # if cuda:
            #     target = target.cuda(async=True)

            # target_var = Variable(target, volatile=(
            #     not train), requires_grad=False)

            # index of the current frame the largest
            # loss = criterion(xq_odom, xq_global, target_var[:, 1:, ...])
            # pose_loss, odom_loss = criterion(
            #     xq_odom, xq_global, target_var[:, 1:, ...])
            #  pose_loss, odom_loss = criterion(
            #     xq_odom, xq_global, target_var[:, 1:, ...])
            pose_loss, odom_loss = criterion(
                xq_odom, xq_global, target_var)

            loss = pose_loss + odom_loss
            if task == 'both':
                optim_loss = loss
            elif task == 'odom':
                optim_loss = odom_loss
            elif task == 'pose':
                optim_loss = pose_loss
            else:
                raise ValueError("Invalid task option")
            if train:
                # SGD step
                optim.learner.zero_grad()
                optim_loss.backward()
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm(
                        model.parameters(), max_grad_norm)
                optim.learner.step()

            return loss.data[0], (xq_odom, xq_global)
        else:
            return 0, (xq_odom, xq_global)

    def train_val(self):
        """
        Function that does the training and validation
        :return:
        """

        for epoch in range(self.start_epoch, self.config.n_epochs):
            self.pose_p = None
            # VALIDATION
            if self.config.do_val and ((epoch % self.config.val_freq == 0) or
                                       (epoch == self.config.n_epochs-1)):
                errs = self.eval(epoch)
                print('Val {:s}: Epoch {:d}'.format(self.experiment,
                                                    epoch))
                print(errs)

            # SAVE CHECKPOINT
            if epoch % self.config.snapshot == 0:
                self.save_checkpoint(epoch)
                print('Epoch {:d} checkpoint saved for {:s}'.
                      format(epoch, self.experiment))

            # ADJUST LR
            if not self.sep_train:
                lr = self.optimizer.adjust_lr(epoch)
            else:
                lr = self.pose_optimizer.adjust_lr(epoch)

            # TRAIN
            self.model.train()
            train_data_time = Logger.AverageMeter()
            train_batch_time = Logger.AverageMeter()
            end = time.time()
            self.pose_p = None
            for batch_idx, (data, target) in enumerate(self.train_loader):
                train_data_time.update(time.time() - end)

                if (not self.sep_train):
                    kwargs = dict(target=target, criterion=self.train_criterion,
                                  optim=self.optimizer, train=True,
                                  max_grad_norm=self.config.max_grad_norm, task='both')

                    loss, _ = self.step_feedfwd(data, self.model, self.config.GPUs > 0,
                                                **kwargs)
                else:
                    # optimize odometry net
                    kwargs = dict(target=target, criterion=self.train_criterion,
                                  optim=self.odom_optimizer, train=True,
                                  max_grad_norm=self.config.max_grad_norm, task='odom')

                    loss, _ = self.step_feedfwd(data, self.model, self.config.GPUs > 0,
                                                **kwargs)
                    # optimize global pose net
                    kwargs['optim'] = self.pose_optimizer
                    kwargs['task'] = 'pose'
                    loss, _ = self.step_feedfwd(data, self.model, self.config.GPUs > 0,
                                                **kwargs)

                train_batch_time.update(time.time() - end)

                if batch_idx % self.config.print_freq == 0:
                    n_iter = epoch*len(self.train_loader) + batch_idx
                    epoch_count = float(n_iter)/len(self.train_loader)
                    print('Train {:s}: Epoch {:d}\t'
                          'Batch {:d}/{:d}\t'
                          'Data Time {:.4f} ({:.4f})\t'
                          'Batch Time {:.4f} ({:.4f})\t'
                          'Loss {:f}\t'
                          'lr: {:f}'.
                          format(self.experiment, epoch, batch_idx, len(self.train_loader)-1,
                                 train_data_time.val, train_data_time.avg, train_batch_time.val,
                                 train_batch_time.avg, loss, lr))
                if batch_idx % self.config.summary_freq == 0:
                    # print(data.cpu().data.numpy())
                    scalar_names_vars = [('loss', loss),
                                         ('sx_abs', self.train_criterion.sx_abs),
                                         ('sq_abs', self.train_criterion.sq_abs),
                                         ('sx_rel', self.train_criterion.sx_rel),
                                         ('sq_rel', self.train_criterion.sq_rel),
                                         ('sx_vo', self.train_criterion.sx_vo),
                                         ('sq_vo', self.train_criterion.sq_vo),
                                         ]
                    image_names_vars = [('input', data[0])]
                    histogram_names_vars = []

                    summarize(self.summary_writer,
                              scalar_names_vars,
                              image_names_vars,
                              histogram_names_vars,
                              batch_idx+epoch*len(self.train_loader))
                end = time.time()

        # Save final checkpoint
        epoch = self.config.n_epochs
        self.save_checkpoint(epoch)
        print('Epoch {:d} checkpoint saved'.format(epoch))
        # if self.config['log_visdom']:
        #     self.vis.save(envs=[self.vis_env])

    def eval(self, epoch):
        """
        Function that does the training and validation
        :return:
        """

        # eval
        # self.model.train()
        with torch.no_grad():
            self.model.eval()
            pred_poses = []  # element: NxTx7
            targ_poses = []  # element: NxTx7

            for batch_idx, (data, target) in enumerate(self.val_loader):

                kwargs = dict(target=target, criterion=self.train_criterion,
                              optim=None, train=False)

                loss, pred = self.step_feedfwd(data, self.model, self.config.GPUs > 0,
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

            qs = [q[:, 3:]/(np.linalg.norm(q[:, 3:], axis=-1, keepdims=True)+1e-12)
                  for q in pred_poses]
            # print(qs[0].shape)
            pred_poses = [np.concatenate((pred_poses[i][:, :3], q), axis=-1)
                          for i, q in enumerate(qs)]
            # print(pred_poses[0].shape)
            # pred_poses = np.vstack(pred_poses)
            pred_poses = np.concatenate(pred_poses, axis=0)

            # only select the current poses
            # targ_poses = [t[1] for t in targ_poses]
            targ_poses = np.vstack(targ_poses)

            t_loss = np.asarray([self.t_criterion(p, t) for p, t in zip(pred_poses[:, :3],
                                                                        targ_poses[:, :3])])
            q_loss = np.asarray([self.q_criterion(p, t) for p, t in zip(pred_poses[:, 3:],
                                                                        targ_poses[:, 3:])])

            errs = OrderedDict({
                "Error in translation(median)": "{:5.3f}".format(np.median(t_loss)),
                "Error in translation(mean)": "{:5.3f}".format(np.mean(t_loss)),
                "Error in rotation(median)": "{:5.3f}".format(np.median(q_loss)),
                "Error in rotation(mean)": "{:5.3f}".format(np.mean(q_loss)),
            })

            if [np.mean(t_loss), np.mean(q_loss)] < [self.t_error_best, self.q_error_best]:
                self.t_error_best = np.mean(t_loss)
                self.q_error_best = np.mean(q_loss)

                filename = osp.join(
                    self.logdir, 'best_epoch_{:03d}.pth.tar'.format(epoch))
                if not self.sep_train:
                    optim_state_dict = self.optimizer.learner.state_dict()
                else:
                    optim_state_dict = copy.deepcopy(
                        self.odom_optimizer.learner.state_dict())
                    optim_state_dict.update(
                        self.pose_optimizer.learner.state_dict())

                checkpoint_dict = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                                   'optim_state_dict': optim_state_dict,
                                   'criterion_state_dict': self.train_criterion.state_dict()}
                torch.save(checkpoint_dict, filename)
            return errs
