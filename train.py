import sys
# sys.path.insert(0, '../')
# sys.path.insert(0, './')
from model import *
from model.trainer import Trainer
from model.vlocnet import VLocNet
from model.optimizer import Optimizer
from model.criterion import Criterion
from dataloaders.composite import MF
import numpy as np
import argparse
import os
import os.path as osp
from torchvision import transforms, models
import torch


parser = argparse.ArgumentParser()
# parser.add_argument("--img_shape", type='str', required=True,
#                     help="The shape of image input: HxW")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--shuffle", type=int, choices=[0, 1], default=0)
parser.add_argument("--num_workers", type=int, default=16,
                    help="The number of threads employed by the data loader")
# optimize
parser.add_argument("--sx", type=float, default=-3,
                    help="Smooth term for translation")
parser.add_argument("--sq", type=float, default=-3,
                    help="Smooth term for rotation")
parser.add_argument("--learn_sxsq", type=int,
                    choices=[0, 1], default=1, help="whether learn sx, sq")
parser.add_argument("--opt_method", type=str,
                    choices=['sgd', 'adam', 'rmsprop'], default='adam', help="The optimization strategy")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Base learning rate.")
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--lr_decay", type=float, default=1,
                    help="The decaying rate of learning rate")

parser.add_argument('--sep_train', type=int, choices=[
                    0, 1], default=1, help='Whether train odometry and global pose network separately')

parser.add_argument('--seed', type=int, default=0,
                    help='')
parser.add_argument('--GPUs', type=int, default=1,
                    help='The number of GPUs employed.')
parser.add_argument('--n_epochs', type=int, default=40,
                    help='The # training epochs')
parser.add_argument('--resume_optim', action='store_true',
                    help='Resume optimization (only effective if a checkpoint is given')

parser.add_argument('--do_val', type=int,
                    choices=[0, 1], default=0, help='Whether do validation when training')
parser.add_argument('--val_freq', type=int, default=1, help='Validation freq')
parser.add_argument('--snapshot', type=int, default=1,
                    help='The snapshot frequency')
parser.add_argument('--max_grad_norm', type=float, default=5,
                    help='The threshold used to clip gradient')
parser.add_argument('--checkpoint_file', type=str, default=None)
# log
parser.add_argument('--logdir', type=str, default='log',
                    help='The directory of logs')
parser.add_argument('--print_freq', type=int, default=20,
                    help='Print frequency')
parser.add_argument('--summary_freq', type=int,
                    default=100, help='The summary frequency.')
# dataloader
parser.add_argument('--data_dir', type=str, required=True,
                    help='The root dir of dataset')
parser.add_argument('--color_jitter', type=float, default=0,
                    help='Data transform')
parser.add_argument('--crop_size', type=int, default=-
                    1, help='Random crop size')
parser.add_argument("--dataset", type=str,
                    choices=['7Scenes', 'RobotCar'], required=True)
parser.add_argument("--scene", type=str, help="Only for the 7Scenes dataset")
parser.add_argument('--skip', type=int, default=10,
                    help='The skip length between adjacent selected frames')
parser.add_argument('--steps', type=int, default=2,
                    help='The number of frames in a clip')
parser.add_argument('--variable_skip', type=int, choices=[0, 1], default=0,
                    help='Whether skip with random lengths')
parser.add_argument('--real', type=int,
                    choices=[0, 1], default=0, help='Whether use ground-truth poses')
# architecture
parser.add_argument('--abs_weight', type=float, default=1,
                    help="The absolute loss weight")
parser.add_argument('--rel_weight', type=float, default=1,
                    help="The relative loss weight")
parser.add_argument('--odom_weight', type=float, default=1,
                    help="The odometry loss weitght")
parser.add_argument('--share_res', type=int, default=3,
                    help="The #shareing level of resnet module")
parser.add_argument('--dropout', type=float, default=0.2,
                    help='The dropout probability')
parser.add_argument('--recur_pose', type=str, default='', choices=['', 'cat', 'add', 'adapt_fusion'],
                    help='The options for recurrent pose.')
parser.add_argument('--recur_train', type=int, default=0, choices=[0, 1],
                    help='Whether use the previous pose prediction as recurrent input.')
parser.add_argument('--pooling_size', type=int, default=1,
                    help='The pooling size of last adpatavgpooling  before the first fc')
parser.add_argument('--model', type=str, default='vlocnet',
                    choices=['vlocnet', 'vlocnet_lstm'], help='The model type.')

args = parser.parse_args()

# parse image's shape
# img_shape = args.image_shape.split('x')
# img_shape = [int(x) for x in img_shape]
# assert(img_shape[0] % 32 == 0 and img_shape[1] % 32 == 0)

# transformers
tforms = [transforms.Resize(256)]
if args.color_jitter > 0:
    assert args.color_jitter <= 1.0
    print('Using ColorJitter data augmentation')
    tforms.append(transforms.ColorJitter(brightness=args.color_jitter,
                                         contrast=args.color_jitter, saturation=args.color_jitter, hue=args.color_jitter/2))
if args.crop_size > 0:
    tforms.append(
        transforms.RandomCrop(min([256, args.crop_size]))
    )

tforms.append(transforms.ToTensor())
# stats_file = osp.join('datasets/data/7Scenes/fire/', 'stats.txt')
# stats = np.loadtxt(stats_file)
# tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))

# simply normalized with the statistics of pretrained ResNet-50
tforms.append(transforms.Normalize(
    mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])
))

data_transform = transforms.Compose(tforms)
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# dataset
kwargs = dict(scene=args.scene, data_path=args.data_dir, transform=data_transform,
              target_transform=target_transform, seed=args.seed)
kwargs = dict(kwargs, dataset=args.dataset, skip=args.skip, steps=args.steps,
              variable_skip=args.variable_skip)
train_set = MF(train=True, real=args.real, **kwargs)
val_set = MF(train=False, real=args.real, **kwargs)

# model
if(args.recur_pose):
    assert(args.crop_size == 224)
if(args.model == 'vlocnet'):
    model = vlocnet.VLocNet(share_levels_n=args.share_res,
                            dropout=args.dropout, recur_pose=args.recur_pose, pooling_size=args.pooling_size)
elif(args.model == 'vlocnet_lstm'):
    model = vlocnet_lstm.VLocNet(share_levels_n=args.share_res,
                                 dropout=args.dropout, recur_pose=args.recur_pose, pooling_size=args.pooling_size)


# criterion

if(args.model == 'vlocnet'):
    train_criterion = criterion.Criterion(
        sx=args.sx,
        sq=args.sq,
        abs_weight=args.abs_weight,
        rel_weight=args.rel_weight,
        odom_weight=args.odom_weight,
        learn_smooth_term=args.learn_sxsq
    )
    val_criterion = Criterion()

elif(args.model == 'vlocnet_lstm'):
    train_criterion = criterion_lstm.Criterion(
        sx=args.sx,
        sq=args.sq,
        abs_weight=args.abs_weight,
        rel_weight=args.rel_weight,
        odom_weight=args.odom_weight,
        learn_smooth_term=args.learn_sxsq
    )
    val_criterion = criterion_lstm.Criterion()
    # optimizer


optim_configs = {
    'sep_train': args.sep_train,
    'method': args.opt_method,
    'base_lr': args.lr,
    'weight_decay': args.weight_decay
}

# param_list = [{'params': model.parameters()}]
# param_list.append({'params': [train_criterion.sx, train_criterion.sq]})
# optimizer = Optimizer(params=param_list, method=args.opt_method, base_lr=args.lr,
#                       weight_decay=args.weight_decay)

# config_name = args.config_file.split('/')[-1]
# config_name = config_name.split('.')[0]

# exp_name
experiment_name = '{:s}_{:s}'.format(args.dataset, args.scene)


# trainer
trainer = Trainer(model=model, optimizer_config=optim_configs, train_criterion=train_criterion, train_dataset=train_set,
                  val_dataset=val_set, config=args, resume_optim=args.resume_optim, val_criterion=val_criterion)

trainer.train_val()
