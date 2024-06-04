import sys
sys.path.append('/home/cwzhang/project/mrc_gan')
import os
# Set TORCH_NCCL_BLOCKING_WAIT to 1 to enable more descriptive errors and potentially prevent the hanging.
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
# Increase the timeout, for example to 24 hours:
os.environ["NCCL_ALLREDUCE_TIMEOUT"] = "86400000"

from argparse import ArgumentParser, Namespace
import torch
from torch import nn
import gc
import wandb
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from gan import GeneratorNestedUNet, Discriminator
from misc import runName, EpochPrintCallback, LearningRateMonitorCallback
from make_dataloader import FakeDataset, GAN_Train_Dataset, GAN_Val_Dataset

torch.set_float32_matmul_precision('high')

import datetime
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 0))  # Bind to a free port provided by the host.
free_port = s.getsockname()[1]  # Get the port number
s.close()

os.environ['MASTER_PORT'] = str(free_port)
os.environ['RANK'] = "0"
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=86400))



class GAN(L.LightningModule):
    def __init__(self,
                 data_shape: tuple, # (1, 32, 32, 32)
                 lrG: float,
                 lrD: float,
                 alpha: float,
                 **kwargs):
        super(GAN, self).__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False
                
        # Generator
        self.generator = GeneratorNestedUNet(in_channels=data_shape[0], out_channels=data_shape[0])
        self.discriminator = Discriminator(in_channels=data_shape[0], num_classes=1)
        # normal initialization        
        self.generator.apply(self.weights_init_normal)
        self.discriminator.apply(self.weights_init_normal)

    def forward(self, x):
        return self.generator(x)

    @staticmethod
    def adversarial_loss(y_pred, y_target):
        return nn.BCEWithLogitsLoss()(y_pred, y_target)
    
    @staticmethod
    def voxelwise_loss(y_pred, y_target):
        return nn.SmoothL1Loss()(y_pred, y_target)    
    
    @staticmethod
    def weights_init_normal(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.InstanceNorm3d): # we don't set weights for InstanceNorm3d layer (affine=False)
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
           
    def training_step(self, batch):
        x, y = batch
        batch_size = x.shape[0]
        
        optimizer_G, optimizer_D = self.optimizers()
        
        # make ground truth labels
        real_label = torch.ones((batch_size, 1), dtype=torch.float, device=self.device, requires_grad=False)
        fake_label = torch.zeros((batch_size, 1), dtype=torch.float, device=self.device, requires_grad=False)
        
        ###################
        # train Generator #
        ###################
        self.toggle_optimizer(optimizer_G)
        
        gen_imgs = self.generator(x)
        
        voxel_loss = self.voxelwise_loss(gen_imgs, y)
        genadv_loss = self.adversarial_loss(self.discriminator(gen_imgs), real_label)
        loss_G = voxel_loss + self.hparams.alpha * genadv_loss 
        
        optimizer_G.zero_grad()
        self.manual_backward(loss_G)
        optimizer_G.step()
        self.untoggle_optimizer(optimizer_G)
    
        #######################
        # train Discriminator #
        #######################
        self.toggle_optimizer(optimizer_D)
        
        real_loss = self.adversarial_loss(self.discriminator(y), real_label)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake_label)    
        loss_D = (real_loss + fake_loss) / 2
        
        optimizer_D.zero_grad()
        self.manual_backward(loss_D)
        optimizer_D.step()
        self.untoggle_optimizer(optimizer_D)
        
        self.log_dict(
            {
            'loss_G': loss_G,
            'loss_D': loss_D,
            'loss_voxelL1': voxel_loss,
            'loss_genadvBCE': genadv_loss,
            'loss_realD': real_loss,
            'loss_fakeD': fake_loss,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
    
    # validation step
    def validation_step(self, batch):
        x, y = batch
        batch_size = x.shape[0]
        
        gen_imgs = self.generator(x)
        
        # make ground truth labels
        real_label = torch.ones((batch_size, 1), dtype=torch.float, device=self.device, requires_grad=False)
        fake_label = torch.zeros((batch_size, 1), dtype=torch.float, device=self.device, requires_grad=False)
        
        voxel_loss = self.voxelwise_loss(gen_imgs, y)
        genadv_loss = self.adversarial_loss(self.discriminator(gen_imgs), real_label)
        loss_G = voxel_loss + self.hparams.alpha * genadv_loss 
        
        real_loss = self.adversarial_loss(self.discriminator(y), real_label)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake_label)    
        loss_D = (real_loss + fake_loss) / 2
        
        self.log_dict(
            {
            # 'epoch': self.current_epoch,
            'val_loss_G': loss_G,
            'val_loss_D': loss_D,
            'val_loss_voxelL1': voxel_loss,
            'val_loss_genadvBCE': genadv_loss,
            'val_loss_realD': real_loss,
            'val_loss_fakeD': fake_loss,
            'val_loss': loss_G + loss_D, # only for EarlyStopping
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )   
        
    def configure_optimizers(self):
        optimizer_G = torch.optim.NAdam(self.generator.parameters(), lr=self.hparams.lrG)
        optimizer_D = torch.optim.NAdam(self.discriminator.parameters(), lr=self.hparams.lrD)
        
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, 
                                                               patience=15, eps=1e-8)
        scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, 
                                                               patience=15, eps=1e-8)
        return [
            {"optimizer": optimizer_G, "lr_scheduler": scheduler_G, "monitor": "val_loss_G"},
            {"optimizer": optimizer_D, "lr_scheduler": scheduler_D, "monitor": "val_loss_D"},
        ]
        
    def on_train_epoch_end(self):
        # clear memory
        torch.cuda.empty_cache() # clear gpu
        gc.collect() # clear cpu
    
    def on_validation_epoch_end(self):
        # Step the schedulers
        val_loss_G = self.trainer.logged_metrics['val_loss_G']
        val_loss_D = self.trainer.logged_metrics['val_loss_D']
        for lr_scheduler in self.lr_schedulers():
            lr_scheduler.step(val_loss_G)
            lr_scheduler.step(val_loss_D)

        # clear memory
        torch.cuda.empty_cache()
        gc.collect()
    

def main(args: Namespace) -> None:
    
    # # Load dataloader
    # # Fake data
    # dataset = FakeDataset(num_samples=100, img_size=(32, 32, 32), batch_size=args.batch_size, num_workers=args.num_workers)
    # train_dataloader, val_dataloader = dataset.create_dataloader()
    
    train_data = GAN_Train_Dataset(train_dir=args.train_dir, cube_text=args.cube_text)
    val_data = GAN_Val_Dataset(val_dir=args.val_dir, cube_text=args.cube_text)
    
    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True
    )
    val_dataloader = DataLoader(
        val_data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False
    )
    
    # Initialize model
    model = GAN(
        data_shape=tuple(train_dataloader.dataset[0][0].shape),
        lrG=args.lrG,
        lrD=args.lrD,
        alpha=args.alpha,
        batch_size=args.batch_size, 
    )
    
    # Make callbacks
    fpath, now_str = runName(args.save_ckpt)
    
    logger = WandbLogger(
    project='GAN',
    save_dir=fpath, 
    name=now_str, 
    )
    lr_monitor = LearningRateMonitorCallback()
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=args.patience, 
        verbose=False
        )
    checkpoints = ModelCheckpoint(
        dirpath=fpath,
        filename='{epoch}-{val_loss_G:.4f}-{val_loss_D:.4f}',
        # monitor='val_loss',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=8,
        )
    epoch_print = EpochPrintCallback()

    # Train model
    trainer = L.Trainer(
        accelerator="gpu", 
        strategy='ddp_find_unused_parameters_true',
        devices=args.gpus, 
        max_epochs=args.epochs, 
        log_every_n_steps=1, 
        logger=logger,
        enable_progress_bar=False,
        callbacks=[
            epoch_print,
            lr_monitor,
            # early_stop,
            checkpoints,
            ],
        )
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Finish any existing WandB run
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--lrG', type=float, default=1e-4)
    parser.add_argument('--lrD', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_ckpt', type=str, default='/results/')
    parser.add_argument('--cube_text', type=str, default='../data')
    parser.add_argument('--train_dir', type=str, default='../data/TrainValData/train_gan_cube_data')
    parser.add_argument('--val_dir', type=str, default='../data/TrainValData/val_gan_cube_data')
    
    args = parser.parse_args()

    main(args)
    
    