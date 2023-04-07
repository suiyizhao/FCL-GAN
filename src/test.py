import torch

from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from utils import *
from options import TestOptions
from models import BUM
from datasets import ImgDataset, ImgLoader

print('---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------')
opt = TestOptions().parse()

single_dir = opt.outputs_dir + '/' + opt.experiment + '/single'
multiple_dir = opt.outputs_dir + '/' + opt.experiment + '/multiple'
clean_dir(single_dir, delete=opt.save_image)
clean_dir(multiple_dir, delete=opt.save_image)

print('---------------------------------------- step 2/4 : data loading... ------------------------------------------------')
print('testing data loading...')
test_dataset = ImgDataset(data_source=opt.data_source, mode='test')
test_dataloader = ImgLoader(test_dataset, batch_size=1, num_workers=1)
print('successfully loading validating pairs. =====> qty:{}'.format(len(test_dataset)))

print('---------------------------------------- step 3/4 : model defining... ----------------------------------------------')
model = BUM().cuda()

model.load_state_dict(torch.load(opt.pretrained_dir + '/' + opt.model_name))

print('---------------------------------------- step 4/4 : testing... ----------------------------------------------------')   
def main():
    model.eval()
    
    psnr_meter = AverageMeter()
    
    for i, (imgs, gts) in enumerate(test_dataloader):
        real_b, real_s = imgs.cuda(), gts.cuda()
        
#         split_real_b = torch.cat(torch.chunk(real_b, chunks=4, dim=2),0)
#         split_real_b = torch.cat(torch.chunk(split_real_b, chunks=4, dim=3),0)
#         with torch.no_grad():
# #             fake_s = model.forward_G_B2S(real_b)
#             split_fake_s = model.forward_G_B2S(split_real_b)
#         split_fake_s = torch.cat(torch.chunk(split_fake_s, chunks=4, dim=0),3)
#         fake_s = torch.cat(torch.chunk(split_fake_s, chunks=4, dim=0),2)
        
        with torch.no_grad():
            fake_s = model.forward_G_B2S(real_b)

        cur_psnr = get_metrics(fake_s, real_s) / fake_s.shape[0]
        psnr_meter.update(get_metrics(fake_s, real_s), fake_s.shape[0])
        
        print('Iter: {} PSNR: {:.4f}'.format(i, cur_psnr))
        
        if opt.save_image:
            save_image(fake_s, single_dir + '/' + str(i).zfill(4) + '.png')
            save_image(real_b, multiple_dir + '/' + str(i).zfill(4) + '_b.png')
            save_image(fake_s, multiple_dir + '/' + str(i).zfill(4) + '_r.png')
            save_image(real_s, multiple_dir + '/' + str(i).zfill(4) + '_s.png')
        
    print('Average PSNR: {:.4f}'.format(psnr_meter.average()))
    
if __name__ == '__main__':
    main()
    