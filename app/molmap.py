import os
import sys
import numpy as np
import mrcfile
import subprocess
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


# Make simulation map
def molmap(pdb, output_mrc, res=2.0, normalize=True, verbose=False):
    # Execute ChimeraX molmap with resolution=2.0
    result = subprocess.run([CHIMERAX_PATH, '--nogui', 
                            '--cmd', 
                            f'open {pdb}; \
                            molmap #1 {res}; \
                            vol #2 step 1 ; \
                            save {output_mrc}_molmap.mrc #2; \
                            exit'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)

    # std mean normalization
    if normalize:
        # Read the map
        simdata, _, simmap_header = read_mrc(f'{output_mrc}_molmap.mrc')
        # Normalize
        norm_data = normalize_map(simdata)
        # Write the normalized map
        write_mrc(f'{output_mrc}_molmap.mrc', norm_data, simmap_header)
            
    if verbose:
        print(result.stdout)
        sys.stdout.flush()
        
    print(f'MolMap saved to {output_mrc}_molmap.mrc')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Molmap')
    parser.add_argument('--pdb', type=str, help='Input pdb file')
    parser.add_argument('--output_mrc', type=str, help='Output mrc file')
    parser.add_argument('--res', type=float, default=2.0, help='Resolution of the map')
    parser.add_argument('--normalize', default=True, help='Normalize the output map')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print verbose output')
    args = parser.parse_args()
    
    molmap(
        pdb=args.pdb,
        output_mrc=args.output_mrc,
        res=args.res,
        normalize=args.normalize,
        verbose=args.verbose,
    )
    
    print(f'MolMap saved to {args.output_mrc}_molmap.mrc')