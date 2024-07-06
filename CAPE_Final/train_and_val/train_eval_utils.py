# Description: This file contains the training and evaluation functions for the model.

# Import necessary libraries
import torch
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

# Define the function to compute the Pearson correlation coefficient, mean absolute error, mean squared error, and Spearman correlation coefficient
def compute_score(prediction, target):
    # inputs are torch.tensor, we transform them to np.array (in cpu)
    prediction = np.nan_to_num(prediction.detach().cpu().numpy().reshape(-1))
    target = target.detach().cpu().numpy().reshape(-1)
    # Compute the Pearson correlation coefficient
    corr = np.corrcoef(prediction, target)
    # Compute the Spearman correlation coefficient
    spear = stats.spearmanr(prediction, target)[0]
    return [corr[0, 1], mean_absolute_error(prediction, target), mean_squared_error(prediction, target), spear]

# Define the function to train the model for one epoch
def train_one_epoch(model, optimizer, criterion, data_loader, device):

    model.train()       # Set the model to training mode
    epoch_loss = 0

    # Iterate over the data loader (batch by batch)
    for i, data in enumerate(data_loader):

        seq_data, pssm_data, labels = data['dna'], data['pssm'], data['labels']
        seq_data = seq_data.to(device)
        pssm_data = pssm_data.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(seq_data, pssm_data)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
        optimizer.step()


        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):

    model.eval()        # Set the model to evaluation mode
    epoch_loss = 0
    eval_acc = 0
    eval_mse = 0
    eval_mae = 0
    R2 = 0
    eval_spear = 0

    # Iterate over the data loader
    for i, data in enumerate(data_loader):
        seq_data, pssm_data, labels = data['dna'], data['pssm'], data['labels']
        seq_data = seq_data.to(device)
        pssm_data = pssm_data.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(seq_data, pssm_data)
        loss = criterion(outputs, labels)

        epoch_loss += loss.item()

        # Compute corr
        ACC, MAE, MSE, spear = compute_score(prediction = outputs, target = labels)
        eval_acc += ACC
        eval_mse += MSE
        eval_mae += MAE
        eval_spear += spear
        out = np.nan_to_num(outputs.detach().cpu().numpy().reshape(-1))
        real = labels.detach().cpu().numpy().reshape(-1)
        R2 += r2_score(real, out)

    # here len(data_loader) is equal to 1 when it is used for the test/val dataset
    return epoch_loss / len(data_loader), eval_acc, R2, eval_mse, eval_mae, eval_spear

