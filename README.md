<div align="center">
# struc2mapGAN: Synthesizing High-resolution Experimental-like Cryo-EM Density Maps with Generative Adversarial Networks
</div>

## About struc2mapGAN
struc2mapGAN is a novel data-driven method that employs a generative adversarial network (GAN) with a U-Net++ architecture as the generator to produce high-resolution experimental-like density maps from molecular structures (PDB files). 

![vida_model](./assets/GAN-architecture.png)   


## Pre-required software

```
Python 3 : 
https://www.python.org/downloads/  

PyTorch 2.2 : 
https://pytorch.org/

UCSF ChimeraX :
https://www.cgl.ucsf.edu/chimerax/download.html
```


## Dependencies
```
numpy==1.26.4
torch==2.2.2
lightning==2.2.3
```

## Installation

```bash
cd struc2mapGAN

pip install -r requirements.txt
```

## Usage
```bash
cd struc2mapGAN/app
```

## Contact

Chenwei Zhang (cwzhang@cs.ubc.ca)