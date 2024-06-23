#!/bin/bash

CKPT=24-0517-132907-last
PDB=8i2h
EMDB=emd_35136


python struc2mapGAN.py --pdb ../example/"$PDB"_ref.pdb --ckpt ../checkpoints/$CKPT.ckpt \
        --output ./output/"$PDB"_$CKPT --batch_size 32


# python struc2mapGAN.py --pdb ../output/"$PDB"_ref.pdb --ckpt ../checkpoints/$CKPT.ckpt \
#         --output ./output/"$PDB"_$CKPT --batch_size 32 --rm_sim --mode pdb2vol