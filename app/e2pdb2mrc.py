# Make sure EMAN2 installed and e2pdb2mrc.py is in the path
# e2pdb2mrc.py --version  -----> EMAN 2.99.47

import os
import sys
import numpy as np
import mrcfile
import subprocess
from copy import deepcopy
import argparse
import pandas as pd


EMAN='e2pdb2mrc.py'
CHIMERAX_PATH='/usr/bin/chimerax'


def normalize_map(map_data):
    if map_data.std() != 0:
        return (map_data - map_data.mean()) / map_data.std()
    else:
        return map_data


def read_mrc(file_path):
    with mrcfile.open(file_path, mode='r') as mrc:
        data = deepcopy(mrc.data)
        voxel_size = mrc.voxel_size
        header = deepcopy(mrc.header)
    return data, voxel_size, header


def write_mrc(file_path, data, header, overwrite=True):
    with mrcfile.new(file_path, overwrite=overwrite) as mrc:
        mrc.set_data(data)
        mrc.header.origin = header.origin
        mrc.header.cella = header.cella
        mrc.header.nxstart = header.nxstart
        mrc.header.nystart = header.nystart
        mrc.header.nzstart = header.nzstart


def resample(chimerax, ref_map, sim_map, sim_map_resample, verbose=False):
    result = subprocess.run([chimerax, '--nogui', 
                            '--cmd', 
                            f'open {ref_map}; \
                            open {sim_map}; \
                            vol #1 #2 step 1 ; \
                            vol resample #2 onGrid #1 gridStep 1; \
                            save {sim_map_resample} #3; \
                            exit'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)
    if verbose:
        print(result.stdout)
        sys.stdout.flush()
        
        print(f'Resampled maps saved to {sim_map_resample}')



# Make simulation map
def e2pdb2mrc(pdb, save_path, res, ref_map, normalize=True, verbose=False):

    x, y, z = mrcfile.open(ref_map, mode='r').data.shape
    
    # Execute ChimeraX molmap with resolution=2.0
    result = subprocess.run([EMAN, 
                             pdb, 
                             f'{save_path}', 
                             '--res', 
                             str(res),
                             '--box',
                             f'{x},{y},{z}',
                             ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)
    # std mean normalization
    if normalize:
        # Read the map
        simdata, _, simmap_header = read_mrc(f'{save_path}')
        # Normalize
        norm_data = normalize_map(simdata)
        # Write the normalized map
        write_mrc(f'{save_path}', norm_data, simmap_header)
            
    if verbose:
        print(result.stdout)
        sys.stdout.flush()
        
        print(f'MolMap saved to {save_path}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='e2pdb2mrc')
    parser.add_argument('--pdb', type=str, help='Input pdb file')
    parser.add_argument('--output_mrc', type=str, help='Output mrc file')
    parser.add_argument('--res', type=float, default=2.0, help='Resolution of the map')
    parser.add_argument('--normalize', default=True, help='Normalize the output map')
    parser.add_argument("--ref_map", default=False, help="Path to a reference map in MRC format")
    parser.add_argument('--verbose', action='store_true', default=False, help='Print verbose output')
    args = parser.parse_args()
    
    
    # sim_map = e2pdb2mrc(
    #             pdb=args.pdb,
    #             save_path=args.output_mrc,
    #             res=args.res,
    #             ref_map=args.ref_map,
    #             normalize=args.normalize,
    #             verbose=args.verbose,
    #         )
    # # # other resample way
    # # directory = os.path.dirname(sim_map)
    # # sim_map_name = os.path.basename(sim_map).split('.')[0]
    # # gan_map_name = f'{sim_map_name}_gan'
    # # sim_map_resample = os.path.join(directory, f'{sim_map_name}_resample.mrc')
    # # resample(CHIMERAX_PATH, args.ref_map, sim_map, sim_map_resample, verbose=False)
    # print(f'e2pdb2mrc saved to {args.output_mrc}_e2pdb2mrc.mrc')
    
    
    df_csv = pd.read_csv('../paper_benchmark/inference_data_raw.csv', dtype=str)
    emd_list = df_csv['EMID']
    pdb_list = df_csv['PDBID']
    
    for i in range(len(pdb_list)):
                
        pdb = f'../paper_benchmark/test_exp_data/{pdb_list[i]}_ref.pdb'
        output_mrc = f'../paper_benchmark/test_gan_data/eman_new/{pdb_list[i]}_e2pdb2mrc.mrc'
        ref_map = f'../paper_benchmark/test_exp_data/emd_{emd_list[i]}.map'
        res = 2.0
        
        # sim_map = e2pdb2mrc(
        #         pdb=pdb,
        #         save_path=output_mrc,
        #         res=res,
        #         ref_map=ref_map,
        #         normalize=args.normalize,
        #         verbose=args.verbose,
        #     )
        sim_map = output_mrc
        
        # # other resample way
        directory = os.path.dirname(sim_map)
        sim_map_name = os.path.basename(sim_map).split('.')[0]
        sim_map_resample = os.path.join(directory, f'{sim_map_name}_resample.mrc')
        resample(CHIMERAX_PATH, ref_map, sim_map, sim_map_resample, verbose=False)        
    
        print(f'PDB-{pdb_list[i]} | EMDB-{emd_list[i]}: e2pdb2mrc saved to {output_mrc}')
        sys.stdout.flush()
            
    
    