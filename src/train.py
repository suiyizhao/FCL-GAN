import time
import torch
import random
import itertools

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from utils import *
from options import TrainOptions
from models import BUM
from losses import CycleLoss, ContrastiveLoss, L_TV, GanLoss, DLoss
from datasets import ImgDataset, ImgLoader

print('---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------')
opt = TrainOptions().parse()

set_random_seed(opt.seed, deterministic=False)

models_dir, log_dir, train_images_dir, val_images_dir = prepare_dir(opt.results_dir, opt.experiment, delete=(not opt.resume))

writer = SummaryWriter(log_dir=log_dir)

print('---------------------------------------- step 2/5 : data loading... ------------------------------------------------')
print('training data loading...')
train_dataset = ImgDataset(data_source=opt.data_source, mode='train', crop=256, random_resize=720)
train_dataloader = ImgLoader(train_dataset, batch_size=opt.train_bs, num_workers=opt.num_workers)
print('successfully loading training pairs. =====> qty:{} bs:{}'.format(len(train_dataset),opt.train_bs))

print('validating data loading...')
val_dataset = ImgDataset(data_source=opt.data_source, mode='val')
val_dataloader = ImgLoader(val_dataset, batch_size=opt.val_bs, num_workers=opt.num_workers)
print('successfully loading validating pairs. =====> qty:{} bs:{}'.format(len(val_dataset),opt.val_bs))

print('---------------------------------------- step 3/5 : model defining... ----------------------------------------------')
model = BUM().cuda()
print_para_num(model)
print_para_num(model.G_B2S)
print_para_num(model.D_B)

print('---------------------------------------- step 4/5 : requisites defining... -----------------------------------------')
# Losses
criterion_cycle = CycleLoss()
criterion_ctst = ContrastiveLoss()
criterion_tv = L_TV()
criterion_gan = GanLoss(gan_type=opt.gan_type)
criterion_d = DLoss(gan_type=opt.gan_type)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(model.G_B2S.parameters(), model.G_S2B.parameters()), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(model.D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_S = torch.optim.Adam(model.D_S.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# Learning rate update schedulers
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, [20, 40, 60, 80], 0.5)
scheduler_D_B = torch.optim.lr_scheduler.MultiStepLR(optimizer_D_B, [20, 40, 60, 80], 0.5)
scheduler_D_S = torch.optim.lr_scheduler.MultiStepLR(optimizer_D_S, [20, 40, 60, 80], 0.5)

# contrastive buffer
buffer_b = ReplayBuffer(max_size=opt.ctst_bs*opt.train_bs*2)
buffer_s = ReplayBuffer(max_size=opt.ctst_bs*opt.train_bs*2)

for i, (imgs, gts) in enumerate(train_dataloader):
    if not buffer_b.is_full:
        buffer_b.pre_fill(imgs.cuda())
        buffer_s.pre_fill(gts.cuda())
    else:
        break

print('Successfully fill the buffers')
print('---------------------------------------- step 5/5 : training... ----------------------------------------------------')
def main():

    optimal = [0.]
    start_epoch = 1
    if opt.resume:
        state = torch.load(models_dir + '/latest.pkl')
        model.load_state_dict(state['model'])
        start_epoch = state['epoch'] + 1
        optimal = state['optimal']
        print('Resume from epoch %d' % (start_epoch), optimal)
    
    for epoch in range(start_epoch, opt.n_epochs + 1):
        train(epoch, optimal)
        
        if (epoch) % opt.val_gap == 0:
            val(epoch, optimal)
        
    writer.close()
    
def train(epoch, optimal):
    model.train()
    
    max_iter = len(train_dataloader)
        
    iter_D_meter = AverageMeter()
    iter_G_meter = AverageMeter()
    iter_gan_meter = AverageMeter()
    iter_cycle_meter = AverageMeter()
    iter_ctst_meter = AverageMeter()
    iter_tv_meter = AverageMeter()
    iter_timer = Timer()
    
    for i, (imgs, gts) in enumerate(train_dataloader):
        real_b, real_s = imgs.cuda(), gts.cuda()
        cur_batch = real_b.shape[0]
        ctst_b = buffer_b.push_and_pop(real_b, get_num=opt.ctst_bs*cur_batch)
        ctst_s = buffer_s.push_and_pop(real_s, get_num=opt.ctst_bs*cur_batch)
        
        fake_s, fake_b, recon_b, recon_s = model.forward_G(real_b, real_s)
        
        # -----------------------
        #  Train Discriminator
        # -----------------------
        
        # -----------------------
        #  Train Discriminator B
        optimizer_D_B.zero_grad()
        # foward
        fake_b_valid, real_b_valid = model.forward_D_B(fake_b.detach(), real_b)
        # compute loss, backward & update
        loss_D_B = criterion_d(fake_b_valid, real_b_valid)
        loss_D_B.backward()
        optimizer_D_B.step()
        
        # -----------------------
        #  Train Discriminator S
        optimizer_D_S.zero_grad()
        # foward
        fake_s_valid, real_s_valid = model.forward_D_S(fake_s.detach(), real_s)
        # compute loss, backward & update
        loss_D_S = criterion_d(fake_s_valid, real_s_valid)
        loss_D_S.backward()
        optimizer_D_S.step()
            
        loss_D = (loss_D_B + loss_D_S) / 2
        
        # ------------------
        #  Train Generator
        # ------------------
            
        model.freeze_D()
            
        optimizer_G.zero_grad()
        # compute loss
        loss_cycle = (criterion_cycle(recon_b, real_b) + criterion_cycle(recon_s, real_s)) / 2
        loss_ctst = (criterion_ctst(fake_b, real_b, ctst_s) + criterion_ctst(fake_s, real_s, ctst_b)) / 2
        loss_tv = criterion_tv(fake_s)
        loss_gan = (criterion_gan(model.D_B(fake_b)) + criterion_gan(model.D_S(fake_s))) / 2
        loss_G = loss_gan + opt.lambda_cycle * loss_cycle + opt.lambda_ctst * loss_ctst + opt.lambda_tv * loss_tv
        # backward & update
        loss_G.backward()
        optimizer_G.step()
            
        model.unfreeze_D()
        
        # record
        iter_D_meter.update(loss_D.item()*cur_batch, cur_batch)
        iter_G_meter.update(loss_G.item()*cur_batch, cur_batch)
        iter_gan_meter.update(loss_gan.item()*cur_batch, cur_batch)
        iter_cycle_meter.update(loss_cycle.item()*cur_batch, cur_batch)
        iter_ctst_meter.update(loss_ctst.item()*cur_batch, cur_batch)
        iter_tv_meter.update(loss_tv.item()*cur_batch, cur_batch)
        
        # print
        if (i+1) % opt.print_gap == 0:
            print('Training: Epoch[{:0>4}/{:0>4}] Iteration[{:0>4}/{:0>4}] Loss_D: {:.4f} Loss_G: {:.4f} Loss_gan: {:.4f} Loss_cycle: {:.4f} Loss_ctst: {:.4f} Loss_tv: {:.4f} Time: {:.4f}'.format(epoch, opt.n_epochs, i + 1, max_iter, iter_D_meter.average(), iter_G_meter.average(), iter_gan_meter.average(), iter_cycle_meter.average(), iter_ctst_meter.average(), iter_tv_meter.average(), iter_timer.timeit()))
            writer.add_scalar('loss_D', iter_D_meter.average(), i+1 + (epoch - 1) * max_iter)
            writer.add_scalar('loss_G', iter_G_meter.average(), i+1 + (epoch - 1) * max_iter)
            writer.add_scalar('loss_gan', iter_gan_meter.average(), i+1 + (epoch - 1) * max_iter)
            writer.add_scalar('loss_cycle', iter_cycle_meter.average(), i+1 + (epoch - 1) * max_iter)
            writer.add_scalar('loss_ctst', iter_ctst_meter.average(), i+1 + (epoch - 1) * max_iter)
            writer.add_scalar('loss_tv', iter_tv_meter.average(), i+1 + (epoch - 1) * max_iter)
            iter_D_meter.reset()
            iter_G_meter.reset()
            iter_gan_meter.reset()
            iter_cycle_meter.reset()
            iter_ctst_meter.reset()
            iter_tv_meter.reset()
            
            # save image
            grid_real_b = make_grid(real_b, nrow=opt.train_bs, normalize=True)
            grid_fake_s = make_grid(fake_s, nrow=opt.train_bs, normalize=True)
            grid_real_s = make_grid(real_s, nrow=opt.train_bs, normalize=True)
            grid_fake_b = make_grid(fake_b, nrow=opt.train_bs, normalize=True)
            
            img_comp = torch.cat((grid_real_b, grid_fake_s, grid_real_s, grid_fake_b), 1)
            save_image(img_comp, train_images_dir + '/img_epoch_{:0>4}_iter_{:0>4}.png'.format(epoch, i+1))
            
    writer.add_scalar('lr', scheduler_G.get_last_lr()[0], epoch)
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'optimal': optimal}, models_dir + '/latest.pkl')
    
    scheduler_G.step()
    scheduler_D_B.step()
    scheduler_D_S.step()
    
def val(epoch, optimal):
    model.eval()
    
    print(''); print('Validating...', end=' ')

    psnr_meter = AverageMeter()
    timer = Timer()
    
    for i, (imgs, gts) in enumerate(val_dataloader):
        real_b, real_s = imgs.cuda(), gts.cuda()
        
        with torch.no_grad():
            fake_s = model.forward_G_B2S(real_b)
        
        psnr_meter.update(get_metrics(fake_s, real_s), real_b.shape[0])
        
        if i == 0:
            if epoch == opt.val_gap:
                save_image(real_b, val_images_dir + '/img_epoch_{:0>4}_iter_{:0>4}.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
                save_image(real_s, val_images_dir + '/gt_epoch_{:0>4}_iter_{:0>4}.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
            save_image(fake_s, val_images_dir + '/restored_epoch_{:0>4}_iter_{:0>4}.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
        
    print('Epoch[{:0>4}/{:0>4}] PSNR: {:.4f} Time: {:.4f}'.format(epoch, opt.n_epochs, psnr_meter.average(), timer.timeit())); print('')
    
    if optimal[0] < psnr_meter.average():
        optimal[0] = psnr_meter.average()
        torch.save(model.state_dict(), models_dir + '/optimal_{:.2f}_epoch_{:0>4}.pkl'.format(optimal[0], epoch))

    writer.add_scalar('psnr', psnr_meter.average(), epoch)
    
    torch.save(model.state_dict(), models_dir + '/epoch_{:0>4}.pkl'.format(epoch))
    
    
if __name__ == '__main__':
    main()
    