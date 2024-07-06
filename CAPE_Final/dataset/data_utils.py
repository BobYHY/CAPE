# import the necessary packages
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Define the class of the promoter dataset
class PromoterDataset(Dataset):
    def __init__(self, text, pssm, labels):
        self.text = torch.from_numpy(text).float()
        self.pssm = torch.from_numpy(pssm).unsqueeze(1) 
        self.labels = torch.from_numpy(labels).unsqueeze(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        content = {'dna': self.text[idx],       # dna sequence of the promoter (kmer2vec representation)
                   'pssm': self.pssm[idx],      # merged CGR matrix of the promoter (the alternative representation of pssm matrix)
                   'labels': self.labels[idx]}  # the expression level of the promoter (after log2 transformation)
        return content

# Define the function to preprocess the data
def data_preprocessing(path_x, path_pssm, path_y, path_w2v, K = 3):
    x = np.load(path_x)
    y = np.load(path_y)
    wv = np.load(path_w2v, allow_pickle = True).item()
    pssm = np.load(path_pssm).astype(np.float32)
    x1 = []
    for i in range(0, len(x)):
        x2 = []
        for j in range(0, len(x[i]) - K + 1):
            x2.append(wv[x[i][j:j + K]])
        x1.append(x2)
    X = np.array(x1).astype(np.float32)     # X is the kmer2vec representation of the promoter
    y = np.array(y).astype(np.float32)      # y is the expression level of the promoter
    y = np.log2(y + 1e-5)                   # log2 transformation of the expression level
    return X, pssm, y

# Define the function to get the k-fold dataset (for task1)
def get_k_fold_dataset(path_x, path_pssm, path_y, path_w2v, Fold, State = 0):
    X, pssm, y = data_preprocessing(path_x, path_pssm, path_y, path_w2v)
    X_used, y_used, pssm_used = X[:11884], y[:11884], pssm[:11884]      # only the first 11884 samples are belonged to the task1 dataset
    kf = KFold(n_splits = Fold, shuffle = True, random_state = State)
    fold_data = []
    # Split the dataset into k folds
    for train_index, val_index in kf.split(X_used):
        X_train, X_val = X_used[train_index], X_used[val_index]
        y_train, y_val = y_used[train_index], y_used[val_index]
        pssm_train, pssm_val = pssm_used[train_index], pssm_used[val_index]

        train_dataset = PromoterDataset(text = X_train, pssm = pssm_train, labels = y_train)
        val_dataset = PromoterDataset(text = X_val, pssm = pssm_val, labels = y_val)

        fold_data.append((train_dataset, val_dataset))
    
    return fold_data


# Define the function to get the k-fold test dataset (for task2)
def get_k_fold_test_dataset(path_x, path_pssm, path_y, path_w2v, Fold, State = 0):
    X, pssm, y = data_preprocessing(path_x, path_pssm, path_y, path_w2v)
    kf = KFold(n_splits = Fold, shuffle = True, random_state = State)
    fold_data = []
    # Split the dataset into k folds
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        pssm_train, pssm_val = pssm[train_index], pssm[val_index]
        train_dataset = PromoterDataset(text = X_train, pssm = pssm_train, labels = y_train)
        val_dataset = PromoterDataset(text = X_val, pssm = pssm_val, labels = y_val)
        fold_data.append((train_dataset, val_dataset))
    return fold_data



