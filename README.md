# DBpedia: Text Classification using Conv Nets

## Overview
This repo contains the code for text classification on DBpedia dataset.
It consists of text preprocessing steps, custom data loading in pytorch and training of CNN architecture
described in (https://arxiv.org/abs/1510.03820)

## Steps to run the code
1. Clone the repository.
1. Download the DBPedia dataset and extract train.csv, valid.csv and test.csv files to dataset folder
2. Run python preprocess.py. This generates numpy files of preprocessed features and labels and stores them in 'npy' folder
3. Execute CUDA_VISIBLE_DEVICES=0 python main.py --dataroot="/ ". This begins the training
