#!/bin/bash

# CKPT=24-0517-132907-last  # Previous good: 24-0513-150226-epoch55     #Recent good: 24-0517-132907-epoch127/last
PDB=3j9s
CKPTNAME=24-0522-003921
OUTPUT_DIR="./output/${PDB}_$CKPTNAME"


if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
fi



# for i in $(seq 7 8 143);
for i in $(seq 0 1 13);
do
       python mrcGAN.py --pdb ../output/"$PDB"_ref.pdb --ckpt ../checkpoints/$CKPTNAME/"$CKPTNAME-epoch""$i".ckpt \
        --output $OUTPUT_DIR/"$PDB"_"$CKPTNAME-epoch""$i" --batch_size 32 --rm_sim
done

# python mrcGAN.py --pdb ../output/"$PDB"_ref.pdb --ckpt ../checkpoints/$CKPTNAME/$CKPTNAME-last.ckpt \
#         --output $OUTPUT_DIR/"$PDB"_$CKPTNAME-last --batch_size 32 --rm_sim