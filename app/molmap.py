import os
import sys
import numpy as np
import mrcfile
import subprocess
import pandas as pd
from copy import deepcopy
import argparse


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
def molmap(pdb, output_mrc, res, normalize=True, verbose=False):
    # Execute ChimeraX molmap with resolution
    result = subprocess.run([CHIMERAX_PATH, '--nogui', 
                            '--cmd', 
                            f'open {pdb}; \
                            molmap #1 {res}; \
                            vol #2 step 1; \
                            save {output_mrc} #2; \
                            exit'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)

    # std mean normalization
    if normalize:
        # Read the map
        simdata, _, simmap_header = read_mrc(output_mrc)
        # Normalize
        norm_data = normalize_map(simdata)
        # Write the normalized map
        write_mrc(output_mrc, norm_data, simmap_header)
            
    if verbose:
        print(result.stdout)
        sys.stdout.flush()
        
    print(f'MolMap saved to {output_mrc}')
    
    return f'{output_mrc}'



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Molmap')
    parser.add_argument('--pdb', type=str, help='Input pdb file')
    parser.add_argument('--output_mrc', type=str, help='Output mrc file')
    parser.add_argument('--res', type=float, default=2.0, help='Resolution of the map')
    parser.add_argument('--normalize', default=True, help='Normalize the output map')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print verbose output')
    args = parser.parse_args()
    
    # molmap(
    #     pdb=args.pdb,
    #     output_mrc=args.output_mrc,
    #     res=args.res,
    #     normalize=args.normalize,
    #     verbose=args.verbose,
    # )
    
    # print(f'MolMap saved to {args.output_mrc}_molmap.mrc')
    
     
    df_csv = pd.read_csv('../paper_benchmark/inference_data.csv', dtype=str)
    emd_list = df_csv['EMID']
    pdb_list = df_csv['PDBID']
    resolution_list = df_csv['Resolution']
    
    for i in range(len(pdb_list)):
                
        pdb = f'../paper_benchmark/data/raw_map_pdb/{pdb_list[i]}_ref.pdb'
        output_mrc = f'../paper_benchmark/data/molmap_ogres/{pdb_list[i]}_molmap_ogres.mrc'
        ref_map = f'../paper_benchmark/data/raw_map_pdb/emd_{emd_list[i]}.map'
        res = resolution_list[i]
        
        sim_map = molmap(
                pdb=pdb,
                output_mrc=output_mrc,
                res=res,
                normalize=args.normalize,
                verbose=args.verbose,
            )
        
        directory = os.path.dirname(sim_map)
        sim_map_name = os.path.basename(sim_map).split('.')[0]
        sim_map_resample = os.path.join(directory, f'{sim_map_name}_resample.mrc')
        resample(CHIMERAX_PATH, ref_map, sim_map, sim_map_resample, verbose=False) 
        
        # # remove original map
        os.remove(sim_map)       
    
        print(f'PDB-{pdb_list[i]} | EMDB-{emd_list[i]}: molmap saved to {output_mrc}')
        sys.stdout.flush()            
    
    