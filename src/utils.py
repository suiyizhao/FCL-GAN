import os
import time
import torch
import random
import shutil
import numpy as np

from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def set_random_seed(seed, deterministic=False):
    
    '''
    function: Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    '''
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_dir(results_dir, experiment, delete=True):
    
    '''
    prepare needed dirs.
    '''
    
    models_dir = os.path.join(results_dir, experiment, 'models')
    log_dir = os.path.join(results_dir, experiment, 'log')
    train_images_dir = os.path.join(results_dir, experiment, 'images', 'train')
    val_images_dir = os.path.join(results_dir, experiment, 'images', 'val')
    
    clean_dir(models_dir, delete=delete)
    clean_dir(log_dir, delete=delete)
    clean_dir(train_images_dir, delete=delete)
    clean_dir(val_images_dir, delete=delete)
    
    return models_dir, log_dir, train_images_dir, val_images_dir

def clean_dir(path, delete=False, contain=False):
    '''
    if delete is True: if path exist, then delete it's files and folders under it, if not, make it;
    if delete is False: if path not exist, make it.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    elif delete:
        delete_under(path, contain=contain)
        
def delete_under(path, contain=False):
    '''
    delete all files and folders under path
    :param path: Folder to be deleted
    :param contain: delete root or not
    '''
    if contain:
        shutil.rmtree(path)
    else:
        del_list = os.listdir(path)
        for f in del_list:
            file_path = os.path.join(path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def print_para_num(model):
    
    '''
    function: print the number of total parameters and trainable parameters
    '''
    
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters: %d' % total_params)
    print('trainable parameters: %d' % total_trainable_params)

class AverageMeter(object):
    
    """
    Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        
    def average(self):
        return self.sum / self.count
    
class Timer(object):
    
    """
    Computes the times.
    """
    
    def __init__(self, start=True):
        if start:
            self.start()

    def start(self):
        self.time_begin = time.time()

    def timeit(self, auto_reset=True):
        times = time.time() - self.time_begin
        if auto_reset:
            self.start()
        return times
    
def get_metrics(tensor_image1, tensor_image2, psnr_only=True, reduction=False):
    
    '''
    function: given a batch tensor image pair, get the mean or sum psnr and ssim value.
    input:  range:[0,1]     type:tensor.FloatTensor  format:[b,c,h,w]  RGB
    output: two python value, e.g., psnr_value, ssim_value
    '''
    
    if len(tensor_image1.shape) != 4 or len(tensor_image2.shape) != 4:
        raise Excpetion('a batch tensor image pair should be given!')
        
    numpy_imgs = tensor2img(tensor_image1)
    numpy_gts = tensor2img(tensor_image2)
    psnr_value, ssim_value = 0., 0.
    batch_size = numpy_imgs.shape[0]
    for i in range(batch_size):
        if not psnr_only:
            ssim_value += structural_similarity(numpy_imgs[i],numpy_gts[i], multichannel=True, gaussian_weights=True, use_sample_covariance=False)
        psnr_value += peak_signal_noise_ratio(numpy_imgs[i],numpy_gts[i])
        
    if reduction:
        psnr_value = psnr_value/batch_size
        ssim_value = ssim_value/batch_size
    
    if not psnr_only:  
        return psnr_value, ssim_value
    else:
        return psnr_value

def tensor2img(tensor):
    
    '''
    function: transform a tensor image to a numpy image
    input:  range:[0,1]     type:tensor.FloatTensor  format:[b,c,h,w]  RGB
    output: range:[0,255]    type:numpy.uint8         format:[b,h,w,c]  RGB
    '''
    
    tensor = tensor*255
    tensor = tensor.permute([0, 2, 3, 1])
    if tensor.device != 'cpu':
        tensor = tensor.cpu()
    img = np.uint8(tensor.numpy())
    return img
    
class ReplayBuffer:
    def __init__(self, max_size=64):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []
        self.is_full = False
    
    def pre_fill(self, element):
        for i in range(len(element)):
            if len(self.data) < self.max_size:
                self.data.append(element[i])
            if len(self.data) == self.max_size:
                self.is_full = True
                break
    
    def push_and_pop(self, element, get_num):
        for i in range(len(element)):
            if len(self.data) < self.max_size:
                self.data.append(element[i])
            else:
                if random.uniform(0, 1) > 0.5:
                    rdn = random.randint(0, self.max_size - 1)
                    self.data[rdn] = element[i]
                    
        random_list = []
        pop_data = []
        
        for i in range(get_num):
            rdn = random.randint(0, len(self.data) - 1)
            while rdn in random_list:
                rdn = random.randint(0, len(self.data) - 1)
            random_list.append(rdn)
            pop_data.append(self.data[rdn])
            
        return torch.stack(pop_data, 0)