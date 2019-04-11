import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torchvision import transforms, models
from .ResNet import resnet_elu

import numpy as np

# class FC(nn.module):
#     #full connected layer via 1x1 conv
#     def __init__(self):
#         pass
#         self.conv = nn.Conv2d()
#     def forward(self, x):
#         (_, C, H, W) = x.data.size()
#         x = x.view(-1, C, 1,1)
#         x = nn.Conv2d(x,)


class ResEncoder(nn.Module):

    def __init__(self, inplanes=64):
        super(ResEncoder, self).__init__()
        layers = [3, 4, 6, 3]
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.elu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(resnet_elu.Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(
            resnet_elu.Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            resnet_elu.Bottleneck, 256, layers[2], stride=2)

        self.layer4 = self._make_layer(
            resnet_elu.Bottleneck, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class GlobalFinalRes(nn.Module):
    def __init__(self, inplanes):
        super(GlobalFinalRes, self).__init__()
        layers = [3, 4, 6, 3]
        self.inplanes = inplanes
        self.out_planes = 2048
        self.layer4 = self._make_layer(
            resnet_elu.Bottleneck, 512, layers[3], stride=2)
        # self.pose_preprocess = nn.Sequential(nn.Linear(6, num_classes) ) #!!!!

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, pre_pose=None):
        x = self.layer4(x)
        return x


class OdomFinalRes(nn.Module):
    def __init__(self, inplanes):
        super(OdomFinalRes, self).__init__()
        layers = [3, 4, 6, 3]
        self.inplanes = inplanes
        self.out_planes = 2048
        self.layer4 = self._make_layer(
            resnet_elu.Bottleneck, 512, layers[3], stride=2)
        # self.pose_preprocess = nn.Sequential(nn.Linear(6, num_classes) ) #!!!!

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, pre_pose=None):
        x = self.layer4(x)
        return x


class ResNetHead(nn.Module):
    def __init__(self):
        super(ResNetHead, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.elu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool(x)

        return x


class ResModule(nn.Module):

    def __init__(self, inplanes, planes, blocks_n, stride, layer_idx,  block=resnet_elu.Bottleneck):
        super(ResModule, self).__init__()
        self.module_name = 'layer'+str(layer_idx)
        self.inplanes = inplanes
        self.planes = planes

        self.resModule = nn.ModuleDict({
            self.module_name:  self._make_layer(
                block, self.planes, blocks_n, stride)
        })

        # self.__dict__.update(
        #     {self.module_name: self._make_layer(
        #         block, self.planes, blocks_n, stride)
        #      }
        # )
        # self.layer = self._make_layer(
        #     block, self.planes, blocks_n, stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.__dict__[self.module_name](x)
        # x = vars(self)[self.module_name](x)
        # x = self.layer(x)
        x = self.resModule[self.module_name](x)
        return x


class VLocNet(nn.Module):
    _inplanes = 64

    def __init__(self, share_levels_n=3, dropout=0.2, recur_pose=True,  block=resnet_elu.Bottleneck):  # fc1_shape
        super(VLocNet, self).__init__()

        layers = [3, 4, 6, 3]
        strides = [1, 2, 2, 2]
        self.block = block
        self.share_levels_n = share_levels_n
        self.dropout = dropout
        self.recur_pose = recur_pose

        self.odom_en1_head = ResNetHead()
        self.odom_en2_head = ResNetHead()  # definitely share
        self.global_en_head = self.odom_en2_head

        # odometry_encoder1
        _layers = []
        self.inplanes = self._inplanes
        for i in range(1, len(layers)):  # layer1..3 corresponding to res2..4 in paper
            planes = 64*2**(i-1)
            _layers.append(
                ResModule(inplanes=self.inplanes, planes=planes,
                          blocks_n=layers[i-1], stride=strides[i-1], layer_idx=i)
            )
            self.inplanes = planes * block.expansion
        self.odom_en1 = nn.Sequential(*_layers)
        # print(self.inplanes, "!!!!")

        # odometry_encoder2 and global_encoder: sharing parts
        _layers = []
        self.inplanes = self._inplanes
        for i in range(1, share_levels_n):  # corresponding to res2..share_levels_n in paper
            planes = 64*2**(i-1)
            _layers.append(
                ResModule(inplanes=self.inplanes, planes=planes,
                          blocks_n=layers[i-1], stride=strides[i-1], layer_idx=i)
            )
            self.inplanes = planes * block.expansion
        self._inplanes_r = self.inplanes  # save the results
        self.odom_en2_share = nn.Sequential(*_layers)
        self.global_en_share = self.odom_en2_share

        # odometry_encoder2: rest parts
        _layers = []
        self.inplanes = self._inplanes_r
        for i in range(share_levels_n, len(layers)):
            planes = 64*2**(i-1)
            _layers.append(
                ResModule(inplanes=self.inplanes, planes=planes,
                          blocks_n=layers[i-1], stride=strides[i-1], layer_idx=i)
            )
            self.inplanes = planes * block.expansion

        self.odom_en2_sep = nn.Sequential(*_layers)

        # global_encoder: rest parts
        _layers = []
        self.inplanes = self._inplanes_r
        for i in range(share_levels_n, len(layers)):
            planes = 64*2**(i-1)
            _layers.append(
                ResModule(inplanes=self.inplanes, planes=planes,
                          blocks_n=layers[i-1], stride=strides[i-1], layer_idx=i)
            )
            self.inplanes = planes * block.expansion

        self.global_en_sep = nn.Sequential(*_layers)

        # odom_final_res:
        self.odom_final_res = ResModule(inplanes=self.inplanes*2,
                                        planes=64*2**(len(layers)-1),
                                        blocks_n=layers[len(layers)-1], stride=strides[len(layers)-1], layer_idx=len(layers))
        self.global_final_res = ResModule(inplanes=self.inplanes if not self.recur_pose else self.inplanes*2,
                                          planes=64*2**(len(layers)-1),
                                          blocks_n=layers[len(layers)-1], stride=strides[len(layers)-1], layer_idx=len(layers))
        # GlobalFinalRes(inplanes=self.inplanes if not self.recur_pose else self.inplanes*2)
        if(self.recur_pose):
            # assert(self.config.crop_size == 224)
            self.previous_pose_fc = nn.Linear(7, 14*14*1024)


######################################################################
        '''
        self.Odometry_en1 = ResEncoder(inplanes=self.inplanes)
        # self.Odometry_en2 = ResEncoder()
        self.Global_en = self.Odometry_en2 = ResEncoder()

        # self.Odometry_final_res, _ = self._make_layer(
        #     resnet_elu.Bottleneck, self.Global_en.inplanes, 512, layers[3], stride=2)
        self.Odometry_final_res = OdomFinalRes(
            inplanes=self.Global_en.inplanes)
        # resnet.ResNet._make_layer(block, 512, layers[3], stride=2)
        self.Global_final_res = GlobalFinalRes(
            inplanes=self.Global_en.inplanes//2)
        '''
        self.odom_avgpool = nn.AdaptiveAvgPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.odom_fc1 = nn.Linear(
            2048, 1024)
        self.odom_fcx = nn.Linear(1024, 3)
        self.odom_fcq = nn.Linear(1024, 4)

        self.global_fc1 = nn.Linear(
            2048, 1024)
        self.global_fcx = nn.Linear(1024, 3)
        self.global_fcq = nn.Linear(1024, 4)

        self.odom_dropout = nn.Dropout(p=self.dropout)
        self.global_dropout = nn.Dropout(p=self.dropout)

    def forward(self, input):
        '''
            input: tuple(images,pose_p)
            images NxTx3xHxW, T=2 for now
            pose_p: NxTx7 ,previous poses except the current one
            return: Nx7, NxTx7
        '''
        """
            s = x.size()
            x = x.view(-1, *s[2:])
            poses = self.mapnet(x)
            poses = poses.view(s[0], s[1], -1)
        """
        images = input[0]
        pose_p = input[1]

        I_c = images[:, 1, ...]
        I_p = images[:, 0, ...]

        s = images.size()
        sp = pose_p.size()
        # print(images.size(), "!!!")
        images = images.view(-1, *s[2:])
        pose_p = pose_p.view(-1, *sp[1:])

        out1 = self.odom_en1_head(I_p)
        out1 = self.odom_en1(out1)  # previous frame

        out2 = self.odom_en2_head(I_c)
        out2 = self.odom_en2_share(out2)  # current frame
        out2 = self.odom_en2_sep(out2)

        out2 = torch.cat([out1, out2], dim=1)
        out2 = self.odom_final_res(out2)
        out2 = self.odom_avgpool(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.odom_fc1(out2)
        out2 = F.elu(out2)
        out2 = self.odom_dropout(out2)

        x_odom = self.odom_fcx(out2)
        q_odom = self.odom_fcq(out2)

        # predict the current and previous pose simultaneously
        out3 = self.global_en_head(images)
        out3 = self.global_en_share(out3)
        out3 = self.global_en_sep(out3)
        if(self.recur_pose):
            recur_features = self.previous_pose_fc(pose_p)
            recur_features = recur_features.view(-1, 1024, 14, 14)
            # print(out3.size(), recur_features.size(), images.size())
            out3 = torch.cat([out3, recur_features], dim=1)
            # print(out3.size())
        out3 = self.global_final_res(out3)

        out3 = self.global_avgpool(out3)
        out3 = out3.view(out3.size(0), -1)
        out3 = self.global_fc1(out3)
        out3 = F.elu(out3)
        out3 = self.global_dropout(out3)

        x_global = self.global_fcx(out3)
        q_global = self.global_fcq(out3)

        x_global = x_global.view(s[0], s[1], -1)
        q_global = q_global.view(s[0], s[1], -1)

        '''
        out1 = self.Odometry_en1(I_p)  # previous frame
        out2 = self.Odometry_en2(I_c)  # current frame
        out2 = torch.cat([out1, out2], dim=1)
        out2 = self.Odometry_final_res(out2)
        out2 = self.Odom_avgpool(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.Odometry_fc1(out2)
        x_odom = self.Odometry_fcx(out2)
        q_odom = self.Odometry_fcq(out2)

        # predict the current and previous pose simultaneously
        out3 = self.Global_en(input)  # out3 = self.Global_en(I_c)

        out3 = self.Global_final_res(out3)
        out3 = self.Global_avgpool(out3)
        out3 = out3.view(out3.size(0), -1)
        out3 = self.Global_fc1(out3)
        x_global = self.Global_fcx(out3)
        q_global = self.Global_fcq(out3)

        x_global = x_global.view(s[0], s[1], -1)
        q_global = q_global.view(s[0], s[1], -1)
        '''
        return torch.cat([x_odom, q_odom], dim=1), torch.cat([x_global, q_global], dim=2)
        # return torch.cat([x_odom, q_odom], dim=1), torch.cat([x_global, q_global], dim=1)
