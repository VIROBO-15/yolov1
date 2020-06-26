import torch.nn as nn
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def expand_cfg(cfg):
    cfg_expanded = []
    for v in cfg:
        if isinstance(v, list):
            times = v[-1]
            for _ in range(times):
                cfg_expanded = cfg_expanded + v[:-1]
        else:
            cfg_expanded.append(v)
    return cfg_expanded


def make_layers(cfg):

    layers = []
    in_channels = 3
    for v in cfg:
        pad = 1 if v[0] == 3 else 0
        if v == 'M':  # Max pool
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif isinstance(v, tuple):
            if len(v) == 3:
                # Conv (kernel_size, out_channels, stride)
                layers += [nn.Conv2d(in_channels, out_channels=v[1], kernel_size=v[0], stride=2, padding=pad)]
            else:
                # Conv (kernel_size, out_channels)
                layers += [nn.Conv2d(in_channels, out_channels=v[1], kernel_size=v[0], padding=pad)]
                layers += [nn.BatchNorm2d(num_features=v[1])]  # BN
                #print('[new] BN is added.')

            layers += [nn.LeakyReLU(0.1)]   # Leaky rectified linear activation
            in_channels = v[1]
    return nn.Sequential(*layers)

class Darknet(nn.Module):

    def __init__(self, opt):
        super(Darknet, self).__init__()
        cfg = [
            (7, 64, 2), 'M',  # 1
            (3, 192), 'M',  # 2
            (1, 128), (3, 256), (1, 256), (3, 512), 'M',  # 3
            [(1, 256), (3, 512), 4], (1, 512), (3, 1024), 'M',  # 4
            [(1, 512), (3, 1024), 2], (3, 1024), (3, 1024, 2),  # 5
            (3, 1024), (3, 1024)  # 6
        ]
        self.opt = opt
        cfg = expand_cfg(cfg)
        self.features = make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(1024 * opt.S * opt.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, opt.S * opt.S * (opt.B * 5 + opt.C)))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), self.opt.S, self.opt.S, (self.opt.B * 5 + self.opt.C) )
        return x




