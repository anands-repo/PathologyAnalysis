#!/bin/bash

python3 celltype_expression.py \
    --expression /home/aramach4/Alzheimers/normalized_gene_expression.txt \
    --group Control \
    --meta_data /home/aramach4/Alzheimers/meta_data.csv \
    --marker_files /home/aramach4/Alzheimers/Astro_GeneList.txt,/home/aramach4/Alzheimers/Endothelia_GeneList.txt,/home/aramach4/Alzheimers/Microglia_GeneList.txt,/home/aramach4/Alzheimers/Neuron_GeneList.txt,/home/aramach4/Alzheimers/Oligodendrocytes_GeneList.txt \
    --output_prefix ./celltype_expression_Control \
    --patient_specific

python3 celltype_expression.py \
    --expression /home/aramach4/Alzheimers/normalized_gene_expression.txt \
    --group AD \
    --meta_data /home/aramach4/Alzheimers/meta_data.csv \
    --marker_files /home/aramach4/Alzheimers/Astro_GeneList.txt,/home/aramach4/Alzheimers/Endothelia_GeneList.txt,/home/aramach4/Alzheimers/Microglia_GeneList.txt,/home/aramach4/Alzheimers/Neuron_GeneList.txt,/home/aramach4/Alzheimers/Oligodendrocytes_GeneList.txt \
    --output_prefix ./celltype_expression_AD \
    --patient_specific

python3 celltype_expression.py \
    --expression /home/aramach4/Alzheimers/normalized_gene_expression.txt \
    --group PSP \
    --meta_data /home/aramach4/Alzheimers/meta_data.csv \
    --marker_files /home/aramach4/Alzheimers/Astro_GeneList.txt,/home/aramach4/Alzheimers/Endothelia_GeneList.txt,/home/aramach4/Alzheimers/Microglia_GeneList.txt,/home/aramach4/Alzheimers/Neuron_GeneList.txt,/home/aramach4/Alzheimers/Oligodendrocytes_GeneList.txt \
    --output_prefix ./celltype_expression_PSP \
    --patient_specific
