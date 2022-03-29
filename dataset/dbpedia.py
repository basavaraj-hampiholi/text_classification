import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

class DBpediaDataset(Dataset):
    def __init__(self, root_dir, npy_path, npy_lbls, transform):       
        self.root_dir = root_dir
        self.data = np.load(root_dir+'/'+npy_path, allow_pickle=True)
        #self.model_labels = np.load(npy_lbls, allow_pickle=True)
        self.model_data = np.array(i[0] for i in self.data)
        self.model_labels = np.array(i[1] for i in self.data)
        

    def __len__(self):
        return len(self.model_data)

    def __getitem__(self, idx):

        text =  self.model_data[idx]
        label = self.model_labels[idx]

        return torch.from_numpy(text), label



# def load_dataset(self, csv_path):
#     csv_data = pd.read_csv(self.root_dir+'/'+csv_path)

#     tokenize = lambda x: x.split()
#     text_field = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
#     label_field = data.LabelField(sequential=False, use_vocab=False)

#     #preprocessed_text = csv_data['text_lemmatized'].apply(lambda x: text_field.preprocess(x))

#     text_field.build_vocab(csv_data, vectors=GloVe(name='6B', dim=300))
#     #label_field.build_vocab(csv_data)

#     word_embeddings = text_field.vocab.vectors
#     vocab_size = len(text_field.vocab)

#     print ("Length of Text Vocabulary: " + str(len(text_field.vocab)))
#     print ("Vector size of Text Vocabulary: ", text_field.vocab.vectors.size())
#     #print ("Label Length: " + str(len(label_field)))

#     dataset = DataFrameDataset(df=csv_data, fields=(('text_lemmatized', text_field),('labels', label_field)))

#     return dataset, text_field, vocab_size, word_embeddings


# class DataFrameDataset(Dataset):
#     def __init__(self, df: pd.DataFrame, fields: list):
#         super(DataFrameDataset, self).__init__([Example.fromlist(list(r), fields) for i, r in df.iterrows()], fields)
            
                     
    
    
