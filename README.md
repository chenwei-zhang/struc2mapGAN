<!-- <div align="center"> -->
# Struc2mapGAN: improving synthetic cryo-EM density maps with generative adversarial networks
<!-- </div> -->


[![arXiv](https://img.shields.io/badge/arXiv-2407.17674-b31b1b.svg)](https://arxiv.org/abs/2407.17674)



## About Struc2mapGAN
Struc2mapGAN is a novel data-driven method that employs a generative adversarial network (GAN) with a U-Net++ architecture as the generator to produce improved experimental-like density maps from molecular structures (PDB files). 

<!-- ![struc2mapGAN](./assets/GAN-architecture.png)    -->
<div align="center">
    <img src="./assets/GAN-architecture.png" alt="struc2mapGAN" width="55%" />
</div>



## Pre-required software

```
Python 3 : 
https://www.python.org/downloads/  

UCSF ChimeraX :
https://www.cgl.ucsf.edu/chimerax/download.html
```


## Dependencies
```
numpy<2.0
torch==2.2.2
lightning==2.2.3
```

## Installation

```bash
# Create conda environment
conda create -n struc2mapGAN python=3.10
conda activate struc2mapGAN

# Clone git repo
git clone https://github.com/chenwei-zhang/struc2mapGAN.git
cd struc2mapGAN

# Install dependencies
pip install -r requirements.txt
```

## Download Checkpoint
```bash
# Download the checkpoint and save to ./ckpt/
wget --content-disposition -P ./ckpt https://osf.io/download/397v2/
```


## Usage
```bash
cd struc2mapGAN/app

# Generate maps
python struc2mapGAN.py --pdb ../example/8i2h_ref.pdb --ckpt ../ckpt/struc2mapGAN.ckpt --output_mrc ../example/8i2h_struc2mapGAN.mrc

# If resample to experimental map's box size
python struc2mapGAN.py --pdb ../example/8i2h_ref.pdb --ckpt ../ckpt/struc2mapGAN.ckpt --output_mrc ../example/8i2h_struc2mapGAN.mrc --ref_map ../example/emd_35136.map
```

### Commands

- --pdb <pdb_path>  Path to the input pdb file
- --ckpt <ckpt_path>  Path to the trained checkpoint
- --output_mrc <output_path> Path to the output map
- --ref_map <map_path> Path to the reference map file if resample to the original experimental map's box size


## Generated Map Library 

Please visit the OSF space at https://osf.io/zcxvs/ to view all the test maps generated using struc2mapGAN.




## Contact

Chenwei Zhang (cwzhang@cs.ubc.ca)


## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
