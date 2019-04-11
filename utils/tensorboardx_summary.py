import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from tensorboardX import SummaryWriter


def summarize(
    writer,
    scalar_names_vars,
    image_names_vars,
    histogram_names_vars,
    n_iter=0
):
    for name, var in scalar_names_vars:
        writer.add_scalar(name, var, n_iter)

    for name, var in image_names_vars:
        writer.add_image(name, var, n_iter)

    for name, var in histogram_names_vars:
        writer.add_histogram(name, var, n_iter)


def graph_summary(model, dummy_input, writer):

    # dummy_input = Variable(torch.rand(1, 3, 224, 224))
    with SummaryWriter() as w:
        model = torchvision.models.alexnet()
        w.add_graph(model, (dummy_input, ))
