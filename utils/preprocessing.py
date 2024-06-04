import os
import sys
import subprocess
import re
import numpy as np
import mrcfile
import pandas as pd
from copy import deepcopy

# Make sure Phenix was insalled and added to the PATH
                
class PreprocessMap():
    def __init__(self, mappath, save_dir, emd_id, PHENIX_MAPBOX='phenix.map_box', CHIMERAX_PATH='/usr/bin/chimerax', verbose=False, rm_interim=True):
        self.phenix_mapbox = PHENIX_MAPBOX
        self.chimerax_path = CHIMERAX_PATH
        self.path = mappath
        self.save_dir = save_dir
        self.verbose = verbose
        self.rm_interim = rm_interim
        self.ogmap_path = os.path.join(self.path, f'emd_{emd_id}.map')
        self.mask_path = os.path.join(self.save_dir, f'{emd_id}_mask')
        self.resample_path = os.path.join(self.save_dir, f'{emd_id}_resample.mrc')
        self.norm_path = os.path.join(self.save_dir, f'{emd_id}_norm.mrc')
        self.sim_path = os.path.join(self.save_dir, f'{emd_id}_sim.mrc')
        self.sim_norm_path = os.path.join(self.save_dir, f'{emd_id}_sim_norm.mrc')
    
    @staticmethod
    def percentileNorm(cleanmap, output_path, percent=99.9):
        # normalize by percentile value
        mrc_data = deepcopy(cleanmap.data)   
        
        ### Percentile normalization ###
        percentile = np.percentile(mrc_data[np.nonzero(mrc_data)], percent)
        mrc_data /= percentile
        # set value < 0 to 0; value > 1 to 1
        mrc_data[mrc_data < 0] = 0
        mrc_data[mrc_data > 1] = 1

        # write to new mrc file
        with mrcfile.new(f'{output_path}', overwrite=True) as mrc:
            mrc.set_data(mrc_data)
            mrc.voxel_size = 1
            mrc.header.origin = cleanmap.header.origin
            mrc.close()
    
    @staticmethod
    def minmaxNorm(cleanmap, output_path):
        # normalize by percentile value
        mrc_data = deepcopy(cleanmap.data)   
        
        ### Min-Max normalization ###
        mrc_data = (mrc_data - np.min(mrc_data)) / (np.max(mrc_data) - np.min(mrc_data))
        
        # write to new mrc file
        with mrcfile.new(f'{output_path}', overwrite=True) as mrc:
            mrc.set_data(mrc_data)
            mrc.voxel_size = 1
            mrc.header.origin = cleanmap.header.origin
            mrc.close()

    def segMap(self, pdb_id):
        # Execute phenix.map_box
        command = [
            self.phenix_mapbox,
            f'pdb_file={self.path}/{pdb_id}_ref.pdb',
            f'ccp4_map_file={self.ogmap_path}',
            'mask_atoms=True',
            f'output_file_name_prefix={self.mask_path}'
            ]
        result = subprocess.run(
                                command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        os.remove(f"{self.mask_path}.pdb")
        
        if self.verbose:
            print(result.stdout)
            sys.stdout.flush()
    
    # Resample original maps to 1.0 A/voxel
    def resampleMap(self):
        # Execute ChimeraX resampling
        result = subprocess.run([self.chimerax_path, '--nogui', 
                                '--cmd', 
                                f'open {self.ogmap_path}; \
                                open {self.mask_path}.ccp4; \
                                vol #1 #2 step 1 ; \
                                vol resample #2 onGrid #1 spacing 1.0 gridStep 1; \
                                vol #3 step 1 ; \
                                save {self.resample_path} #3; \
                                exit'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        if self.verbose:
            print(result.stdout)
            sys.stdout.flush()
            
    
    # Make normalized original map
    def normalizeMap(self):
        # Normalize
        cleanmap = mrcfile.open(f'{self.resample_path}', mode='r')
        self.minmaxNorm(cleanmap, self.norm_path) # min-max normalization
            
        if self.verbose:
            print(f'Normalized map saved as {self.norm_path}')
            sys.stdout.flush()
        
        
    # Make simulation map
    def makeSimMap(self, pdb_id):
        # Execute ChimeraX molmap with resolution=2.0
        result = subprocess.run([self.chimerax_path, '--nogui', 
                                '--cmd', 
                                f'open {self.norm_path}; \
                                open {self.path}/{pdb_id}_ref.pdb; \
                                vol #1 step 1 ; \
                                molmap #2 2.0 onGrid #1; \
                                vol #3 step 1 ; \
                                save {self.sim_path} #3; \
                                exit'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        if self.verbose:
            print(result.stdout)
            sys.stdout.flush()
            
        # Make normalized map
        cleanmap = mrcfile.open(f'{self.sim_path}', mode='r')
        self.minmaxNorm(cleanmap, self.sim_norm_path) # min-max normalization
        
        if self.verbose:
            print(f'Normalized Simulated map saved as {self.sim_norm_path}')
            sys.stdout.flush()

        
    # Process the map
    def process(self, emd_id, pdb_id):
        self.segMap(pdb_id)
        self.resampleMap()
        self.normalizeMap()
        self.makeSimMap(pdb_id)
        
        # delete interim maps
        if self.rm_interim:
            os.remove(f'{self.resample_path}')
            os.remove(f'{self.sim_path}')
            os.remove(f'{self.mask_path}.ccp4')
        
        if self.verbose:
            print(f'Resampled map removed: {self.resample_path}')
            print(f'Simulated map removed: {self.sim_path}')
            sys.stdout.flush()
            
            
        print(f'Processing of EMDB-{emd_id} completed.')
        sys.stdout.flush()
        
    
    # cross-corelation between two maps 
    # [map1: normalized original map, map2: normalized simulated map]
    def crossCorrelation(self, emd_id):
        # Execute ChimeraX measure correaltion
        result = subprocess.run([self.chimerax_path, '--nogui', 
                                '--cmd', 
                                f'open {self.norm_path}; \
                                open {self.sim_norm_path}; \
                                vol #1 step 1; \
                                vol #2 step 1 ; \
                                measure correlation #1 inMap #2; \
                                exit'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        
        match = re.search(r'correlation\s*=\s*([0-9.]+)', result.stdout)
        
        if match:
            correlation_value = float(match.group(1))
            print(f'EMDB-{emd_id}:  cross-correlation = {correlation_value}')
            sys.stdout.flush()
            
        else:
            print('No correlation value found in the message.')
            sys.stdout.flush()
            
        return correlation_value
        
        
def main():
    # load csv
    df_csv = pd.read_csv('../data/train_GAN_data.csv', dtype=str)
    columns_csv = {col: df_csv[col].tolist() for col in df_csv.columns}
    emd = columns_csv['EMID']
    pdb = columns_csv['PDBID']

    # set path
    mappath = '../data/raw_gan_data'
    save_dir = '../data/processed_gan_data'
    # make directories if not exist
    os.makedirs(mappath, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(len(pdb)):
        preprocess = PreprocessMap(mappath=mappath, save_dir=save_dir, emd_id=emd[i], 
                                   verbose=False, rm_interim=True)
        
        preprocess.process(emd[i], pdb[i])
    
    
        ### delete the row of csv if correlation < 0.6 ###
        ### Do this before segementation ###
        # correlation_value = preprocess.crossCorrelation(emd[i])
        # if correlation_value < 0.6:
        #     print(f'  [Removed] EMDB-{emd[i]} from the dataset.')
        #     sys.stdout.flush()
        #     df_csv.drop(df_csv[df_csv['EMID'] == emd[i]].index, inplace=True)
        #     df_csv.to_csv('../data/train_GAN_data.csv', index=False) 


if __name__ == '__main__':
    main()
    