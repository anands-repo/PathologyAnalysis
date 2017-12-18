#!/bin/bash

MARKERFILES="/home/aramach4/Alzheimers/data/Astro_GeneList.txt,/home/aramach4/Alzheimers/data/Endothelia_GeneList.txt,/home/aramach4/Alzheimers/data/Microglia_GeneList.txt,/home/aramach4/Alzheimers/data/Neuron_GeneList.txt,/home/aramach4/Alzheimers/data/Oligodendrocytes_GeneList.txt"

# python3 classifier.py \
#     --expression /home/aramach4/Alzheimers/data/CorrectedGeneCounts.txt \
#     --groups "AD;Control" \
#     --meta_data /home/aramach4/Alzheimers/data/meta_data.csv \
#     --train_test_val "0.9,0.0,0.1" \
#     --classifier_type LR \
#     --num_folds 10 \
#     --marker_files $MARKERFILES \
#     --remove_group_effects
# 
# python3 classifier.py \
#     --expression /home/aramach4/Alzheimers/data/CorrectedGeneCounts.txt \
#     --groups "AD;Control" \
#     --meta_data /home/aramach4/Alzheimers/data/meta_data.csv \
#     --train_test_val "0.9,0.0,0.1" \
#     --classifier_type LR \
#     --num_folds 10 \
#     --marker_files $MARKERFILES
# 
# python3 classifier.py \
#     --expression /home/aramach4/Alzheimers/data/CorrectedGeneCounts.txt \
#     --groups "AD;Control" \
#     --meta_data /home/aramach4/Alzheimers/data/meta_data.csv \
#     --train_test_val "0.9,0.0,0.1" \
#     --classifier_type LR \
#     --num_folds 10

python3 classifier.py \
    --expression /home/aramach4/Alzheimers/data/CorrectedGeneCounts.txt \
    --groups "AD;Control" \
    --meta_data /home/aramach4/Alzheimers/data/meta_data.csv \
    --train_test_val "0.8,0.1,0.1" \
    --classifier_type NN \
    --num_folds 10 \
    --num_hidden 2048 \
    --dropout 0.5 \
    --marker_files $MARKERFILES \
    --remove_group_effects

python3 classifier.py \
    --expression /home/aramach4/Alzheimers/data/CorrectedGeneCounts.txt \
    --groups "AD;Control" \
    --meta_data /home/aramach4/Alzheimers/data/meta_data.csv \
    --train_test_val "0.8,0.1,0.1" \
    --classifier_type NN \
    --num_folds 10 \
    --num_hidden 2048 \
    --dropout 0.5 \
    --marker_files $MARKERFILES

python3 classifier.py \
    --expression /home/aramach4/Alzheimers/data/CorrectedGeneCounts.txt \
    --groups "AD;Control" \
    --meta_data /home/aramach4/Alzheimers/data/meta_data.csv \
    --train_test_val "0.8,0.1,0.1" \
    --classifier_type NN \
    --num_hidden 2048 \
    --dropout 0.5 \
    --num_folds 10
