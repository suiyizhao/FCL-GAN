import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, norm=True, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, norm=True, relu=False)
        )
        
    def forward(self, x):
        return self.main(x) + x
    

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, num_res = 8):
        super(Generator, self).__init__()
        base_channel = 24
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, norm=True, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, norm=True, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, norm=True, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, norm=True, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, norm=True, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, norm=True, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, norm=True, relu=True, stride=1),
        ])

    def forward(self, x):

        z = self.feat_extract[0](x)
        res1 = self.Encoder[0](z)

        z = self.feat_extract[1](res1)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        z = self.feat_extract[3](z)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z = self.feat_extract[4](z)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)

        return torch.sigmoid(z+x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        base_channel = 24
        
        self.model = nn.Sequential(
            *self.discriminator_block(3, base_channel, normalize=False),
            *self.discriminator_block(base_channel, base_channel*2),
            *self.discriminator_block(base_channel*2, base_channel*4),
            *self.discriminator_block(base_channel*4, base_channel*8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(base_channel*8, 1, 4, padding=1)
        )
    
    def forward(self, x):
        
        z = self.model(x)
        
        return z
    
    def discriminator_block(self, in_channel, out_channel, normalize=True):
        
        layers = [nn.Conv2d(in_channel, out_channel, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channel))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers
    
class BUM(nn.Module):
    # Basic unsupervised model
    def __init__(self, num_res=6):
        super(BUM, self).__init__()
        
        self.G_B2S = Generator(num_res = num_res)
        self.G_S2B = Generator(num_res = num_res)
        self.D_B = Discriminator()
        self.D_S = Discriminator()

    def forward_G(self, b, s):
        
        fake_s = self.G_B2S(b)
        fake_b = self.G_S2B(s)
        
        recon_b = self.G_S2B(fake_s)
        recon_s = self.G_B2S(fake_b)
        
        return fake_s, fake_b, recon_b, recon_s
    
    def forward_G_B2S(self, b):
        
        fake_s = self.G_B2S(b)
        
        return fake_s
    
    def forward_D_B(self, fake_b, real_b):
        
        fake_b_valid = self.D_B(fake_b)
        real_b_valid = self.D_B(real_b)
        
        return fake_b_valid, real_b_valid
    
    def forward_D_S(self, fake_s, real_s):
        
        fake_s_valid = self.D_S(fake_s)
        real_s_valid = self.D_S(real_s)
        
        return fake_s_valid, real_s_valid
    
    def freeze_D(self):
        for p in self.D_B.parameters():
            p.requires_grad = False
        for p in self.D_S.parameters():
            p.requires_grad = False
            
    def unfreeze_D(self):
        for p in self.D_B.parameters():
            p.requires_grad = True
        for p in self.D_S.parameters():
            p.requires_grad = True