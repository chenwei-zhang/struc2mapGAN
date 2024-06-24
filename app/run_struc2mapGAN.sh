#!/bin/bash

CKPT=24-0517-132907-last
PDB=8i2h
EMDB=emd_35136



python struc2mapGAN.py --pdb ../example/"$PDB"_ref.pdb --ckpt ../ckpt/"$CKPT".ckpt \
        --output_mrc ../example/"$PDB"_struc2mapGAN.mrc


# resample
python struc2mapGAN.py --pdb ../example/"$PDB"_ref.pdb --ckpt ../ckpt/"$CKPT".ckpt \
        --output_mrc ../example/"$PDB"_struc2mapGAN.mrc --ref_map ../example/"$EMDB".map

