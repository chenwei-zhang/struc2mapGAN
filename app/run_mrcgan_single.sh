#!/bin/bash

# CKPT=24-0522-003921-epoch60  # Previous good: 24-0513-150226-epoch55     #Recent good: 24-0517-132907-epoch127/last
CKPT=24-0517-132907-last
PDB=8i2h


# python mrcGAN.py --pdb ../output/"$PDB"_ref.pdb --ckpt ../checkpoints/$CKPT.ckpt \
#         --output ./output/"$PDB"_$CKPT --batch_size 32


python mrcGAN.py --pdb ../output/"$PDB"_ref.pdb --ckpt ../checkpoints/$CKPT.ckpt \
        --output ./output/"$PDB"_$CKPT --batch_size 32 --rm_sim --mode pdb2vol