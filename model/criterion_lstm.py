import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from torchvision import transforms, models
from .ResNet import resnet_elu
from utils import quaternion_math as qm


class Criterion(nn.Module):
    def __init__(self,
                 sx=-3, sq=-3,
                 abs_weight=1, rel_weight=1, odom_weight=1,
                 learn_smooth_term=True):
        super(Criterion, self).__init__()
        self.sx_abs = nn.Parameter(torch.Tensor(
            [sx]), requires_grad=bool(learn_smooth_term))
        self.sq_abs = nn.Parameter(torch.Tensor(
            [sq]), requires_grad=bool(learn_smooth_term))

        self.sx_rel = nn.Parameter(torch.Tensor(
            [sx]), requires_grad=bool(learn_smooth_term))
        self.sq_rel = nn.Parameter(torch.Tensor(
            [sq]), requires_grad=bool(learn_smooth_term))
        self.sx_vo = nn.Parameter(torch.Tensor(
            [sx]), requires_grad=bool(learn_smooth_term))
        self.sq_vo = nn.Parameter(torch.Tensor(
            [sq]), requires_grad=bool(learn_smooth_term))

        self.abs_weight = abs_weight
        self.rel_weight = rel_weight
        self.odom_weight = odom_weight

        self.loss_func = nn.MSELoss()

    def forward(self, xq_odom, xq_global, xq_gt):
        '''
            xq_odom: Nx(T-1)x7
            xq_global, xq_gt: NxTx7
        '''

        s = xq_odom.size()
        num_poses = s[0]

        x_odom = xq_odom[:, :, :3]
        q_odom = F.normalize(xq_odom[:, :, 3:], dim=-1)

        x_global = xq_global[:, 1:, :3]  # T-1
        q_global = F.normalize(xq_global[:, 1:, 3:], dim=-1)
        x_global_p = xq_global[:, :-1, :3]
        q_global_p = xq_global[:, :-1, 3:]

        x_gt = xq_gt[:, 1:, :3]
        q_gt = xq_gt[:, 1:, 3:]
        x_gt_p = xq_gt[:, :-1, :3]
        q_gt_p = xq_gt[:, :-1, 3:]

        # global
        abs_x_loss = self.loss_func(x_global, x_gt)
        abs_q_loss = self.loss_func(q_global, q_gt)
        rel_x_loss = self.loss_func(x_global-x_global_p, x_gt-x_gt_p)
        rel_q_loss = self.loss_func(
            qm.qmult(qm.qinv(q_global_p), q_global),
            qm.qmult(qm.qinv(q_gt_p), q_gt)
        )

        # odometry
        odom_x_loss = self.loss_func(x_odom, x_gt - x_gt_p)

        odom_q_loss = self.loss_func(q_odom,
                                     qm.qmult(qm.qinv(q_gt_p), q_gt)
                                     )

        abs_global_loss = torch.exp(-self.sx_abs)*(abs_x_loss) + self.sx_abs \
            + torch.exp(-self.sq_abs)*(abs_q_loss) + self.sq_abs
        rel_global_loss = torch.exp(-self.sx_rel)*(rel_x_loss) + self.sx_rel \
            + torch.exp(-self.sq_rel)*(rel_q_loss) + self.sq_rel

        odom_loss = torch.exp(-self.sx_vo)*(odom_x_loss) + self.sx_vo \
            + torch.exp(-self.sq_vo)*odom_q_loss + self.sq_vo

        pose_loss = self.abs_weight*abs_global_loss + self.rel_weight*rel_global_loss
        odom_loss = self.odom_weight * odom_loss

        # loss = pose_loss + odom_loss

        return pose_loss, odom_loss
        # return loss

    # def forward(self, xq_odom, xq_global, xq_gt):
    # '''
    #     xq_odom: Nx(T-1)x7
    #     xq_global, xq_gt: NxTx7
    # '''

    # s = xq_odom.size()
    # num_poses = s[0]

    # x_odom = xq_odom[:, -1:, :3]
    # q_odom = F.normalize(xq_odom[:, -1:, 3:], dim=-1)

    # x_global = xq_global[:, -1:, :3]  # T-1
    # q_global = F.normalize(xq_global[:, -1:, 3:], dim=-1)
    # x_global_p = xq_global[:, -2:-1, :3]
    # q_global_p = xq_global[:, -2:-1, 3:]

    # x_gt = xq_gt[:, -1:, :3]
    # q_gt = xq_gt[:, -1:, 3:]
    # x_gt_p = xq_gt[:, -2:-1, :3]
    # q_gt_p = xq_gt[:, -2:-1, 3:]

    # # global
    # abs_x_loss = self.loss_func(x_global, x_gt)
    # abs_q_loss = self.loss_func(q_global, q_gt)
    # rel_x_loss = self.loss_func(x_global-x_global_p, x_gt-x_gt_p)
    # rel_q_loss = self.loss_func(
    #     qm.qmult(qm.qinv(q_global_p), q_global),
    #     qm.qmult(qm.qinv(q_gt_p), q_gt)
    # )

    # # odometry
    # odom_x_loss = self.loss_func(x_odom, x_gt - x_gt_p)

    # odom_q_loss = self.loss_func(q_odom,
    #                                 qm.qmult(qm.qinv(q_gt_p), q_gt)
    #                                 )

    # abs_global_loss = torch.exp(-self.sx_abs)*(abs_x_loss) + self.sx_abs \
    #     + torch.exp(-self.sq_abs)*(abs_q_loss) + self.sq_abs
    # rel_global_loss = torch.exp(-self.sx_rel)*(rel_x_loss) + self.sx_rel \
    #     + torch.exp(-self.sq_rel)*(rel_q_loss) + self.sq_rel

    # odom_loss = torch.exp(-self.sx_vo)*(odom_x_loss) + self.sx_vo \
    #     + torch.exp(-self.sq_vo)*odom_q_loss + self.sq_vo

    # pose_loss = self.abs_weight*abs_global_loss + self.rel_weight*rel_global_loss
    # odom_loss = self.odom_weight * odom_loss

    # # loss = pose_loss + odom_loss

    # return pose_loss, odom_loss
    # # return loss
