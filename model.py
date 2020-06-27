from network import *
from loss import *
import torch.nn as nn
import torchvision
import torch


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.Darknet = Darknet(opt)
        self.Darknet.apply(weights_init_normal).to(device)
        self.optimizer = torch.optim.Adam(list(self.Darknet.parameters()), lr =opt.lr, betas = (opt.beta1, opt.beta2))

    def forward(self, inp, target):

        self.optimizer.zero_grad()
        gen = self.Darknet(inp)

        loss = calc_loss(gen.clone(), target.clone(), self.opt)

        loss.backward()
        self.optimizer.step()
        return loss.item()






