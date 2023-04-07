import glob
import torch
import random

import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImgDataset(Dataset):
    def __init__(self, data_source, mode, crop=256, random_resize=None):
        if not mode in ['train', 'val', 'test']:
            raise Exception('The mode should be "train", "val" or "test".')
        
        self.random_resize = random_resize
        self.crop = crop
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.img_paths = sorted(glob.glob(data_source + '/' + mode + '/blurry' + '/*/*.*'))
        self.gt_paths = sorted(glob.glob(data_source + '/'  + mode + '/sharp' + '/*/*.*'))
            
        self.img_leading = True if len(self.img_paths) >= len(self.gt_paths) else False
        
    def __getitem__(self, index):
        if self.mode == 'train':
            if self.img_leading:
                img = Image.open(self.img_paths[index % len(self.img_paths)]).convert('RGB')
                gt = Image.open(self.gt_paths[random.randint(0, len(self.gt_paths) - 1)]).convert('RGB')
            else:
                img = Image.open(self.img_paths[random.randint(0, len(self.img_paths) - 1)]).convert('RGB')
                gt = Image.open(self.gt_paths[index % len(self.gt_paths)]).convert('RGB')
        else:
            img = Image.open(self.img_paths[index % len(self.img_paths)]).convert('RGB')
            gt = Image.open(self.gt_paths[index % len(self.gt_paths)]).convert('RGB')

        img = self.transform(img)
        gt = self.transform(gt)
        
        if self.mode == 'train':
            if self.random_resize is not None:
                # random resize
                scale_factor = random.uniform(self.crop/self.random_resize, 1.)
                img = F.interpolate(img.unsqueeze(0), scale_factor=scale_factor, align_corners=False, mode='bilinear', recompute_scale_factor=False).squeeze(0)
                gt = F.interpolate(gt.unsqueeze(0), scale_factor=scale_factor, align_corners=False, mode='bilinear', recompute_scale_factor=False).squeeze(0)
            
            # crop
            h, w = img.size(1), img.size(2)
            offset_h = random.randint(0, max(0, h - self.crop - 1))
            offset_w = random.randint(0, max(0, w - self.crop - 1))

            img = img[:, offset_h:offset_h + self.crop, offset_w:offset_w + self.crop]
            gt = gt[:, offset_h:offset_h + self.crop, offset_w:offset_w + self.crop]
        
            # flip
            # vertical flip
            if random.random() < 0.5:
                idx = [i for i in range(img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                img = img.index_select(1, idx)
                gt = gt.index_select(1, idx)
            # horizontal flip
            if random.random() < 0.5:
                idx = [i for i in range(img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                img = img.index_select(2, idx)
                gt = gt.index_select(2, idx)
        
        return img, gt

    def __len__(self):
        return max(len(self.img_paths), len(self.gt_paths))

def ImgLoader(dataset, batch_size, num_workers):
    shuffle = True if dataset.mode == 'train' else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        
    return dataloader

# # Use the following code when a CPU bottleneck occurs
# class data_prefetcher():
#     def __init__(self, loader):
#         self.loader = iter(loader)
#         self.stream = torch.cuda.Stream()
        
#         self.preload()

#     def preload(self):
#         try:
#             self.next_input, self.next_target = next(self.loader)
#         except StopIteration:
#             self.next_input = None
#             self.next_target = None
#             return
#         with torch.cuda.stream(self.stream):
#             self.next_input = self.next_input.cuda(non_blocking=True)
#             self.next_target = self.next_target.cuda(non_blocking=True)
            
#     def next(self):
#         torch.cuda.current_stream().wait_stream(self.stream)
#         input = self.next_input
#         target = self.next_target
#         self.preload()
#         return input, target