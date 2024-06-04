###################################################################################################
# getting from https://github.com/DrDongSi/Ca-Backbone-Prediction/blob/f921e971ac2bb6e70844619f0f420a6f34dabeba/cnn/map_splitter.py
# This files contains functions used to split a large protein into smallers 64^3 chunks that can
# fit into the training CNN. This is accomplished without any input from the user. When the output
# image is reconstructured only the middle 50^3 region in the image is used to build the output
# protein. This helps eliminate the issue of boundary prediction issues.
# Moritz, Spencer  November, 2018
###################################################################################################

import os
import numpy as np
import math
import mrcfile

'''
box_size = 32 # Expected dimensions to the NNs
core_size = 20 # Core of the image where we don't need to worry about boundry issues
'''


def get_manifest_dimentions(image_shape, core_size=20):
    dimentions = [0, 0, 0]
    dimentions[0] = math.ceil(image_shape[0] / core_size) * core_size
    dimentions[1] = math.ceil(image_shape[1] / core_size) * core_size
    dimentions[2] = math.ceil(image_shape[2] / core_size) * core_size
    
    return dimentions


# Creates a list of boxes (32*32*32)
def create_cube(full_image, box_size=32, core_size=20):
    image_shape = np.shape(full_image)
    padded_image = np.zeros((image_shape[0] + 2 * box_size, 
                             image_shape[1] + 2 *box_size, 
                             image_shape[2] + 2 * box_size
                             ))
    padded_image[box_size:box_size + image_shape[0], 
                 box_size:box_size + image_shape[1], 
                 box_size:box_size + image_shape[2]
                 ] = full_image

    cube_list = []

    start_point = box_size - int((box_size - core_size) / 2)
    cur_x = start_point
    cur_y = start_point
    cur_z = start_point
    
    while cur_z + (box_size - core_size) / 2 < image_shape[2] + box_size:
        next_chunk = padded_image[cur_x:cur_x + box_size, 
                                  cur_y:cur_y + box_size, 
                                  cur_z:cur_z + box_size]
        
        cube_list.append(next_chunk)
        
        cur_x += core_size
        
        if cur_x + (box_size - core_size) / 2 >= image_shape[0] + box_size:
            cur_y += core_size
            cur_x = start_point # Reset
            if cur_y + (box_size - core_size) / 2  >= image_shape[1] + box_size:
                cur_z += core_size
                cur_y = start_point # Reset
                cur_x = start_point # Reset
                
    return cube_list


# Takes the output of the NNs and reconstructs the full dimentionality of the map
def reconstruct_map(cube_list, image_shape, box_size=32, core_size=20):
    extract_start = int((box_size - core_size) / 2)
    extract_end = int((box_size - core_size) / 2) + core_size
    dimentions = get_manifest_dimentions(image_shape)

    reconstruct_image = np.zeros((dimentions[0], dimentions[1], dimentions[2]))
    counter = 0
    
    for z_steps in range(int(dimentions[2] / core_size)):
        for y_steps in range(int(dimentions[1] / core_size)):
            for x_steps in range(int(dimentions[0] / core_size)):
                
                reconstruct_image[
                    x_steps * core_size:(x_steps + 1) * core_size, 
                    y_steps * core_size:(y_steps + 1) * core_size, 
                    z_steps * core_size:(z_steps + 1) * core_size
                    ] = cube_list[counter][
                        extract_start:extract_end, 
                        extract_start:extract_end, 
                        extract_start:extract_end
                        ]
                    
                counter += 1
                
    reconstruct_image = np.array(reconstruct_image, dtype=np.float32)
    reconstruct_image = reconstruct_image[
        :image_shape[0], 
        :image_shape[1], 
        :image_shape[2]
        ]
    
    return reconstruct_image


def write2map_gan(raw_map, nn_pred, save_dir):
    # extract name
    base = os.path.basename(raw_map)
    name = os.path.splitext(base)[0]
    new_map_name = name + '_gan' + '.mrc'
    
    new_map_path = os.path.join(save_dir, new_map_name)
    
    with mrcfile.open(raw_map, 'r') as raw_map:
        raw_voxelsize = raw_map.voxel_size
        raw_gridsize = raw_map.data.shape
        raw_header = raw_map.header
        
    with mrcfile.new(new_map_path, overwrite=True) as new_map:
        new_map.set_data(nn_pred)
        new_map.header.origin = raw_header.origin
        new_map.header.cella = raw_header.cella
        new_map.header.nxstart = raw_header.nxstart
        new_map.header.nystart = raw_header.nystart
        new_map.header.nzstart = raw_header.nzstart
        new_map_voxelsize = new_map.voxel_size
        new_map_gridsize = new_map.data.shape
        
    assert new_map_gridsize ==  raw_gridsize, "The grid size of GAN map mismatched with input sim map"
    assert new_map_voxelsize ==  raw_voxelsize, "The voxel size of GAN map mismatched with input sim map"
    
    return new_map_path
    
        
        