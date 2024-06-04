import os
import time
import pytz
from datetime import datetime
import lightning as L


# File naming
def runName(save_dir):
    # Get UTC time and convert to Pacific Time
    now_utc = datetime.now(pytz.timezone('UTC'))
    now_pst = now_utc.astimezone(pytz.timezone('US/Pacific'))
    now_str = now_pst.strftime('%y-%m%d-%H%M%S')
    # create the log savepath
    fpath = f'{save_dir}/checkpoints/gan_model/{now_str}'
    os.makedirs(fpath, exist_ok=True)
    
    return fpath, now_str      


# Callbacks
class EpochPrintCallback(L.Callback):
    def on_train_start(self, trainer, pl_module, column_width=12):
        self.column_width = column_width
        title = "|"
        title += "epoch".center(self.column_width) + "|"
        title += "time".center(self.column_width) + "|"
        title += "loss_G".center(self.column_width) + "|"
        title += "loss_D".center(self.column_width) + "|"
        title += "val_loss_G".center(self.column_width) + "|"
        title += "val_loss_D".center(self.column_width) + "|"
        self.row = "-" * len(title)
        print(self.row)
        print(title)
        print(self.row)
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.start_time
        output = "\r|"
        output += f"{trainer.current_epoch}".center(self.column_width) + "|"
        output += f"{epoch_time:.2f}".center(self.column_width) + "|"
        output += f"{trainer.logged_metrics['loss_G']:.4f}".center(self.column_width) + "|"
        output += f"{trainer.logged_metrics['loss_D']:.4f}".center(self.column_width) + "|"
        output += f"{trainer.logged_metrics['val_loss_G']:.4f}".center(self.column_width) + "|"
        output += f"{trainer.logged_metrics['val_loss_D']:.4f}".center(self.column_width) + "|"
        print(output)
        print(self.row) 
        
    def on_train_end(self, trainer, pl_module):
        done = "| Training completed on epoch " + str(trainer.current_epoch-1) + " / " + str(trainer.max_epochs-1) + " |"
        row = "-" * len(done)
        print("\n")
        print(row)
        print(done)
        print(row)
        
        
class LearningRateMonitorCallback(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        for i, lr_scheduler_config in enumerate(trainer.lr_scheduler_configs):
            scheduler = lr_scheduler_config.scheduler
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            if i == 0:
                self.log('lr_G', current_lr, sync_dist=True)
            elif i == 1:
                self.log('lr_D', current_lr, sync_dist=True)