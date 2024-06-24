import sys
sys.path.append('../../struc2mapGAN')

import os
from argparse import ArgumentParser, Namespace
import numpy as np
import mrcfile
import torch
import gc
import lightning as L
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.gan import GeneratorNestedUNet, Discriminator
from model.make_dataloader import GAN_Pred_Dataset
from utils.map2cube import reconstruct_map, write2map_gan

torch.set_float32_matmul_precision('high')



class GAN(L.LightningModule):
    def __init__(self,
                 data_shape: tuple = (1, 32, 32, 32)
                 ):
        super(GAN, self).__init__()
        
        self.generator = GeneratorNestedUNet(in_channels=data_shape[0], out_channels=data_shape[0])
        self.discriminator = Discriminator(in_channels=data_shape[0], num_classes=1)

    def forward(self, x):
        return self.generator(x)


def inference(map, ckpt, batch_size, num_workers, save_dir):
    # Load data
    mapdata = mrcfile.open(map, mode='r').data
    mapshape = mapdata.shape
    
    pred_dataset = GAN_Pred_Dataset(map=mapdata)
    pred_dataloader = DataLoader(pred_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Load model
    model = GAN.load_from_checkpoint(ckpt)
    # model = GAN()
    
    
    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Inference
    model.eval()  # Set the model to evaluation mode
    all_cubes = []
    
    with torch.no_grad():
        for batch in tqdm(pred_dataloader):
            images = batch
            images = images.to(device)
            generated_images = model(images).detach().cpu()        
            all_cubes.append(generated_images)
            
    concatenated_cubes = torch.cat(all_cubes, dim=0)
    pred_data =  concatenated_cubes.squeeze(dim=1).numpy()
    
    # Cubes to the map
    pred_map = reconstruct_map(pred_data, image_shape=mapshape, box_size=32, core_size=20)

    assert pred_map.shape == mapshape, "The shape of the predicted map is not the same as the input map."
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    # Write the enhanced map
    new_map_path = write2map_gan(raw_map=map, nn_pred=pred_map, save_dir=save_dir)

    return new_map_path

        
        
if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('--map', type=str, required=True, help='Path to the map to be enhanced')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--save_dir', type=str, default='.', help='Path to save the enhanced map')
    args = parser.parse_args()
    
    new_map_path = inference(
                        args.map,
                        args.ckpt,
                        args.batch_size,
                        args.num_workers,
                        args.save_dir
                        )
    
    print(f'The GAN modified map saved to {new_map_path}')

