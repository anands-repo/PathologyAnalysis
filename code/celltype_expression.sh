#!/bin/bash

MARKERFILES="/home/aramach4/Alzheimers/data/Astro_GeneList.txt,/home/aramach4/Alzheimers/data/Endothelia_GeneList.txt,/home/aramach4/Alzheimers/data/Microglia_GeneList.txt,/home/aramach4/Alzheimers/data/Neuron_GeneList.txt,/home/aramach4/Alzheimers/data/Oligodendrocytes_GeneList.txt"

python3 celltype_expression.py \
    --expression /home/aramach4/Alzheimers/data/CorrectedGeneCounts.txt \
    --group Control \
    --meta_data /home/aramach4/Alzheimers/data/meta_data.csv \
    --marker_files $MARKERFILES \
    --output_prefix ./celltype_expression_Control

python3 celltype_expression.py \
    --expression /home/aramach4/Alzheimers/data/CorrectedGeneCounts.txt \
    --group AD \
    --meta_data /home/aramach4/Alzheimers/data/meta_data.csv \
    --marker_files $MARKERFILES \
    --output_prefix ./celltype_expression_AD

python3 celltype_expression.py \
    --expression /home/aramach4/Alzheimers/data/CorrectedGeneCounts.txt \
    --group PSP \
    --meta_data /home/aramach4/Alzheimers/data/meta_data.csv \
    --marker_files $MARKERFILES \
    --output_prefix ./celltype_expression_PSP
