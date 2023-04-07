import argparse

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # ---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------
        self.parser.add_argument("--seed", type=int, default=1234, help="random seed")
        self.parser.add_argument("--resume", action='store_true', help="if specified, resume the training")
        self.parser.add_argument("--results_dir", type=str, default='../results', help="path of saving models, images, log files")
        self.parser.add_argument("--experiment", type=str, default='experiment', help="name of experiment")
        
        # ---------------------------------------- step 2/5 : data loading... ------------------------------------------------
        self.parser.add_argument("--data_source", type=str, default='', required=True, help="dataset root")
        self.parser.add_argument("--train_bs", type=int, default=4, help="size of the training batches")
        self.parser.add_argument("--val_bs", type=int, default=2, help="size of the validating batches")
        self.parser.add_argument("--ctst_bs", type=int, default=8, help="size of the negative batches when contrastive learning")
        self.parser.add_argument("--num_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
        
        # ---------------------------------------- step 3/5 : model defining... ------------------------------------------------
        
        
        # ---------------------------------------- step 4/5 : requisites defining... ------------------------------------------------
        self.parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
        self.parser.add_argument("--n_epochs", type=int, default=80, help="number of epochs of training")
        
        self.parser.add_argument("--gan_type", type=str, default='lsgan', help="the type of gan, optional: lsgan, vanilla")
        
        # ---------------------------------------- step 5/5 : training... ------------------------------------------------
        self.parser.add_argument("--print_gap", type=int, default=40, help="the gap between two print operations, in iteration")
        self.parser.add_argument("--val_gap", type=int, default=1, help="the gap between two validations, also the gap between two saving operation, in epoch")
        
        self.parser.add_argument("--lambda_cycle", type=float, default=10.0, help="cycle loss weight")
        self.parser.add_argument("--lambda_ctst", type=float, default=0.1, help="contrastive loss weight")
        self.parser.add_argument("--lambda_tv", type=float, default=0.1, help="tv norm weight")
        
    def parse(self, show=True):
        opt = self.parser.parse_args()
        
        if show:
            self.show(opt)
        
        return opt
    
    def show(self, opt):
        
        args = vars(opt)
        print('************ Options ************')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('************** End **************')
        
class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # ---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------
        self.parser.add_argument("--outputs_dir", type=str, default='../outputs', help="path of saving models, images, log files")
        self.parser.add_argument("--experiment", type=str, default='experiment', help="name of experiment")
        
        # ---------------------------------------- step 2/4 : data loading... ------------------------------------------------
        self.parser.add_argument("--data_source", type=str, default='', required=True, help="dataset root")
        
        # ---------------------------------------- step 3/4 : model defining... ------------------------------------------------
        self.parser.add_argument("--pretrained_dir", type=str, default='../pretrained', help="pretrained model root")
        self.parser.add_argument("--model_name", type=str, default='', required=True, help="name of the model to be loaded")
        
        # ---------------------------------------- step 4/4 : testing... ------------------------------------------------
        self.parser.add_argument("--save_image", action='store_true', help="if specified, save image when testing")
        
    def parse(self, show=True):
        opt = self.parser.parse_args()
        
        if show:
            self.show(opt)
        
        return opt
    
    def show(self, opt):
        
        args = vars(opt)
        print('************ Options ************')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('************** End **************')
    