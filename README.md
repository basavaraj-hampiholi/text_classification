# DBpedia: Sentence Classification using Conv Nets

## Overview
This repo contains the code for text classification on DBpedia dataset.
It consists of text preprocessing steps, dataloading and training of CNN architecture
described in (https://arxiv.org/abs/1510.03820)

## Steps to run the code

1. Download the DBPedia dataset to dataset folder
2. Run python preprocess.py. This generates numpy files for preprocessed features and labels
3. Execute CUDA_VISIBLE_DEVICES=0 python main.py. This begins the training
