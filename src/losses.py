import os
import torch
import torch.nn as nn
import torchvision.models as models

class CycleLoss():
    def __init__(self):
        self.criterion = nn.L1Loss()
    
    def __call__(self, img1, img2):
        loss = self.criterion(img1, img2)
        return loss
    
class IdentityLoss():
    def __init__(self):
        self.criterion = nn.L1Loss()
    
    def __call__(self, img1, img2):
        loss = self.criterion(img1, img2)
        return loss

class PerceptualLoss():
    def __init__(self, path, layer=28):
        vgg_path = path + '/vgg19.pth'
        if not os.path.exists(vgg_path):
            self.model = models.vgg19(pretrained=True)
        else:
            self.model = models.vgg19(pretrained=False)
            self.model.load_state_dict(torch.load(vgg_path))
            
        self.model = self.model.cuda()
        self.model = self.model.features[:layer]
        self.model.eval()
        
        self.criterion = nn.L1Loss()
    
    def __call__(self, img1, img2):
        feature1 = self.model(img1)
        feature2 = self.model(img2)
        loss = self.criterion(feature1, feature2)
        return loss

class L_TV(nn.Module):
    
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()

    def forward(self,x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv

class ContrastiveLoss():
    def __init__(self, T=0.07, window_size=64):
        self.T = T
        self.FDM = FourierDiscriminantMeta(shift=False, one_side=False)
        self.criterion = nn.CrossEntropyLoss()
        self.similarity_criterion = nn.MSELoss()
        self.blockpool = OverLapRatioPool2d(window_size=window_size)
        
    def __call__(self, img, pos, batch_neg):
        img = self.FDM.rgb2gray(img)
        pos = self.FDM.rgb2gray(pos)
        batch_neg = self.FDM.rgb2gray(batch_neg)

        img_meta = self.FDM.visualize(self.FDM.get_meta(img), binarization=True)[:,:,64:192,64:192]
        pos_meta = self.FDM.visualize(self.FDM.get_meta(pos), binarization=True)[:,:,64:192,64:192]
        batch_neg_meta = self.FDM.visualize(self.FDM.get_meta(batch_neg), binarization=True)[:,:,64:192,64:192]
        
        blocked_img_meta = self.blockpool(img_meta)
        blocked_pos_meta = self.blockpool(pos_meta)
        blocked_batch_neg_meta = self.blockpool(batch_neg_meta)
        blocked_batch_neg_meta = torch.stack(torch.chunk(blocked_batch_neg_meta, chunks=blocked_img_meta.shape[0], dim=0), 0)
        
        pred = list()
        for i in range(blocked_img_meta.shape[0]):
            
            single_pred = list()
            single_pred.append((1. - self.similarity_criterion(blocked_img_meta[i], blocked_pos_meta[i])) / self.T)

            for blocked_neg_meta in blocked_batch_neg_meta[i]:
                single_pred.append((1. - self.similarity_criterion(blocked_img_meta[i], blocked_neg_meta)) / self.T)
            
            single_pred = torch.cuda.FloatTensor(single_pred)
                
            pred.append(single_pred)
            
        pred = torch.stack(pred, dim=0)
        label = torch.zeros([pred.shape[0]], dtype=torch.int64).cuda()
        loss = self.criterion(pred, label)
        return loss
    
class GanLoss():
    def __init__(self, gan_type):
        self.gan_type = gan_type
        if self.gan_type == 'vanilla':
            self.criterion = nn.BCELoss()
        elif self.gan_type == 'lsgan':
            self.criterion = nn.MSELoss()
        
    def __call__(self, fake_valid):
        
        all1 = torch.ones_like(fake_valid).cuda()
        if self.gan_type == 'vanilla':
            loss = self.criterion(torch.sigmoid(fake_valid), all1)
        elif self.gan_type == 'lsgan':
            loss = self.criterion(fake_valid, all1)
        else:
            raise Exception('no such type of gan')
            
        return loss

class DLoss():
    def __init__(self, gan_type):
        self.gan_type = gan_type
        if self.gan_type == 'vanilla':
            self.criterion = nn.BCELoss()
        elif self.gan_type == 'lsgan':
            self.criterion = nn.MSELoss()
    
    def __call__(self, fake_valid, real_valid):
        
        all0 = torch.zeros_like(fake_valid).cuda()
        all1 = torch.ones_like(real_valid).cuda()
        if self.gan_type == 'vanilla':
            fake_loss = self.criterion(torch.sigmoid(fake_valid), all0)
            real_loss = self.criterion(torch.sigmoid(real_valid), all1)
            loss = (fake_loss + real_loss) / 2
        elif self.gan_type == 'lsgan':
            fake_loss = self.criterion(fake_valid, all0)
            real_loss = self.criterion(real_valid, all1)
            loss = (fake_loss + real_loss) / 2
        else:
            raise Exception('no such type of gan')
            
        return loss

class FourierDiscriminantMeta(object):
    def __init__(self, shift=True, one_side=False):
        self.shift = shift
        self.one_side = one_side
    
    def get_meta(self, img):
        # input: 4-D img (b,1,h,w)
        
        if self.one_side:
            f= torch.fft.rfft2(img, dim=(-2,-1))
        else:
            f= torch.fft.fft2(img, dim=(-2,-1))
        
        if not self.shift:
            return f
        f = torch.fft.fftshift(f, dim=(-2,-1))
        
        return f
    
    def visualize(self, f, save_path=None, binarization=False, normalize=False):
        f_img = 20 * torch.log(torch.abs(f))
        if binarization:
            all_one = torch.ones_like(f_img)
            all_zero = torch.zeros_like(f_img)
            f_img = torch.where(f_img>0, all_one, all_zero)
            
        if save_path is not None:
            save_image(f_img, save_path, normalize=normalize)
        
        return f_img
    
    def inverse_meta(self, f):
        # input: 4-D fft (b,1,h,w)
        
        if self.shift:
            f = torch.fft.ifftshift(f, dim=(-2,-1))
            
        if self.one_side:
            f= torch.fft.irfft2(f, dim=(-2,-1))
        else:
            f= torch.fft.ifft2(f, dim=(-2,-1))
        
        return torch.abs(f)
    
    def rgb2gray(self, tensor):
        '''
        function: Convert tensor image to grayscale
        input:   format:[b,c,h,w]  RGB
        output:  format:[b,1,h,w]  gray
        '''

        b, _, h, w = tensor.shape
        R = tensor[:, 0, :, :]
        G = tensor[:, 1, :, :]
        B = tensor[:, 2, :, :]

        gray = 0.299*R+0.587*G+0.114*B

        return gray.view(b, 1, h, w)

class OverLapRatioPool2d(nn.Module):
    def __init__(self, window_size=64):
        super(OverLapRatioPool2d, self).__init__()
        self.window_size = window_size
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, feature):
        _, _, h, w = feature.shape
        feature = 1 - feature
        rows = torch.cat(torch.chunk(feature, chunks=h//self.window_size, dim=2), dim=0)
        blocks = torch.cat(torch.chunk(rows, chunks=w//self.window_size, dim=3), dim=0)
        blocks = self.pool(blocks)
        rows = torch.cat(torch.chunk(blocks, chunks=w//self.window_size, dim=0), dim=3)
        out = torch.cat(torch.chunk(rows, chunks=h//self.window_size, dim=0), dim=2)
        return out