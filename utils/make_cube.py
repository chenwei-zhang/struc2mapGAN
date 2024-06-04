import os
import sys
import numpy as np
import mrcfile
import torchio
import pandas as pd
from map2cube import create_cube


# save cubes of each map
def get_manifest(input_map, target_map, box_size, core_size):
    
    with mrcfile.open(input_map, mode='r') as input_data:
        input_data = input_data.data
        
    with mrcfile.open(target_map, mode='r') as target_data:
        target_data = target_data.data

    training_transform = torchio.Compose(
        [
            torchio.RandomAnisotropy(
                downsampling=1.5, image_interpolation='bspline', p=0.25
            ),
            torchio.RandomBlur((0, 0.5), p=0.25),
            torchio.RandomNoise(std=0.1, p=0.25)
        ]
    )
    input_data = torchio.ScalarImage(tensor=input_data[None, ...])
    input_data = training_transform(input_data)
    input_data = input_data.tensor.squeeze().numpy().astype(np.float32)
        
    input_cube = np.array(
        create_cube(input_data, box_size=box_size, core_size=core_size),
        dtype = np.float32,
    )
    
    target_cube = np.array(
        create_cube(target_data, box_size=box_size, core_size=core_size),
        dtype = np.float32,
    )
    
    return input_cube, target_cube


def save_cubes(map_dir, save_dir, emd_list, box_size, core_size):
    
    for emd in emd_list:
        input_map = os.path.join(map_dir, f'{emd}_sim_norm.mrc')
        target_map = os.path.join(map_dir, f'{emd}_norm.mrc')
        
        input_cube, target_cube = get_manifest(input_map, target_map, box_size=box_size, core_size=core_size)
        
        for i in range(len(input_cube)):
            fname = os.path.join(save_dir, f'{emd}_cb_{i}') 
            np.savez_compressed(
                file = fname,
                input_cube = input_cube[i],
                target_cube = target_cube[i],
            )
        
        print(f"Done EMDB-{emd}")
        sys.stdout.flush()        


def write2txt(txtfile, dirpath):
    names = os.listdir(dirpath)
    with open(txtfile, 'w') as f:
        for name in names:
            f.write(name + '\n')

"""
Run Saving
"""
os.makedirs('../data/train_gan_cube_data', exist_ok=True)
os.makedirs('../data/val_gan_cube_data', exist_ok=True)

emd_list = pd.read_csv('../data/train_GAN_data.csv', dtype=str)['EMID'].tolist()  ## val_GAN_data.csv
save_cubes(map_dir='../data/processed_gan_data', save_dir='../data/train_gan_cube_data', 
           emd_list=emd_list, box_size=32, core_size=20)  ## val_gan_cube_data

# # write cube names to txt file
# dirpath = '../data/train_gan_cube_data'
# txtfile = '../data/train_gan_cubes.txt'
# write2txt(txtfile, dirpath)