import numpy as np
import subprocess
from copy import deepcopy
from skimage.metrics import structural_similarity as ssim


'''
Pearson Correlaion Coefficient (PCC) calculation
'''
def calculate_pcc(map1, map2, percentile=90):
    # Ensure the maps are numpy arrays
    map1 = deepcopy(map1)
    map2 = deepcopy(map2)
    
    threshold1 = np.percentile(map1[np.nonzero(map1)], percentile)
    threshold2 = np.percentile(map2[np.nonzero(map2)], percentile)
    
    map1[map1 < threshold1] = 0
    map2[map2 < threshold2] = 0

    # Normalize maps min-max
    map1 = (map1 - np.min(map1)) / (np.max(map1) - np.min(map1))
    map2 = (map2 - np.min(map2)) / (np.max(map2) - np.min(map2))
    
    # Calculate the means of each map
    mean_map1 = np.mean(map1)
    mean_map2 = np.mean(map2)

    # Calculate PCC
    pcc = np.sum((map1 - mean_map1) * (map2 - mean_map2)) / np.sqrt(np.sum((map1 - mean_map1)**2) * np.sum((map2 - mean_map2)**2))

    return pcc



'''
Structural Similarity Index (SSIM) calculation
'''

def calculate_ssim(map1, map2, percentile=90):

    map1 = deepcopy(map1)
    map2 = deepcopy(map2)

    threshold1 = np.percentile(map1[np.nonzero(map1)], percentile)
    threshold2 = np.percentile(map2[np.nonzero(map2)], percentile)

    map1[map1 < threshold1] = 0
    map2[map2 < threshold2] = 0
    
    # min-max normalize maps
    map1 = (map1 - np.min(map1)) / (np.max(map1) - np.min(map1))
    map2 = (map2 - np.min(map2)) / (np.max(map2) - np.min(map2))
    
    ssim_value = ssim(map1, map2, data_range=map2.max() - map2.min())
    
    return ssim_value




'''
Cosine Similarity calculation (Used in ChimeraX)
map1: path to the reference experimental map
map2: path to the generated/simulated map
'''

def calculate_cosine(map1, map2, chimerax='/usr/bin/chimerax'):
    result = subprocess.run([chimerax, 
                             '--nogui',
                             '--cmd',
                             f'open {map1}; \
                            open {map2}; \
                            vol #1 #2 step 1; \
                            measure correlation #1 inMap #2;\
                            exit'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    
    # Decode and process output
    output = result.stdout.decode()
    correlations = [line for line in output.split('\n') if 'correlation' in line]

    # Split the string into parts by commas
    parts = correlations[-1].split(',')

    # Extract correlation values
    correlation = parts[0].split('=')[1].strip()
    correlation_about_mean = parts[1].split('=')[1].strip()

    return float(correlation), float(correlation_about_mean)
