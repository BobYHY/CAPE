from model.promoter_model import PrompterModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from dataset.data_utils import get_k_fold_dataset, get_k_fold_dataset
from dataset.matrix import matrix_generation
from train_and_val.train_eval_utils import train_one_epoch, evaluate
from sklearn.metrics import r2_score

# set the random seed to make the experiments reproduciable
ss = 3407
torch.manual_seed(ss) 
torch.cuda.manual_seed_all(ss)
torch.backends.cudnn.deterministic = True

# Define the function to count the number of trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# configs
in_dim = 100
embed_dim = 32
out_dim = 1
depth_transformer = 2
heads_transformer = 8
n_epochs = 100
lr = 8e-4 
lr_steps = [10, 30] 
lr_gamma = 0.1
batch_size = 128
patience = 100

train_valid_name = "Ecoli"
testname1 = "trc" 
Update_Train = False
Update_Test1 = False
K_FOLD = 5
best_weight_dir1 = './best_Ecoli.pth'
best_weight_dir2 = './best_trc.pth'

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # Set the device to GPU if available
    CGR_N = 20      # The size of the CGR matrix

    # Generate the CGR matrix for the training and validation dataset if needed
    if Update_Train == True:
        matrix_generation(train_valid_name, CGR_N)
    if Update_Test1 == True:
        matrix_generation(testname1, CGR_N)

    # Get the k-fold dataset for task1 (Ecoli dataset)
    Task1_fold = get_k_fold_dataset(path_x = './data/' + train_valid_name + '_seq.npy', path_pssm = './data/' + train_valid_name + '_matrix.npy',
                                             path_y = './data/' + train_valid_name + '_y.npy', path_w2v = './data/word2vec.npy', Fold = K_FOLD)
    Task1_acc = [0] * K_FOLD
    Task1_R2 = [0] * K_FOLD
    Task1_mse = [0] * K_FOLD
    Task1_mae = [0] * K_FOLD
    Task1_spear = [0] * K_FOLD

    # Train the model and evaluate the performance for each fold
    for F, (train_dataset, val_dataset) in enumerate(Task1_fold):
        print(f'Fold {F+1}')

        # Create the data loader for training and validation
        train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
        val_loader = DataLoader(dataset = val_dataset, batch_size = len(val_dataset), shuffle = False)

        # Create the model
        model = PrompterModel(input_dim = in_dim,
                          embedding_dim = embed_dim,
                          depth_transformer = 2,
                          heads_transformer = 8,
                          dim_head_transformer = 64,
                          attn_dropout_transformer = 0.1,
                          ff_dropout_transformer = 0.1,
                          dropout_CNN = 0.2,
                          mat_size = CGR_N
                          )
        
        model = model.to(device)
        print(f'The model has {count_parameters(model):,} trainable parameters')
        print(next(model.parameters()).device)

        # Define the optimizer, loss function, learning rate scheduler, and other monitors
        optimizer = optim.Adam(model.parameters(), lr = lr)
        criterion = nn.L1Loss()
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = lr_steps, gamma = lr_gamma)
        monitor1 = -1
        monitor2 = -1
        monitor3 = -1
        monitor4 = -1
        monitor5 = -1
        bad_epochs = 0

        # Train the model
        for epoch in range(n_epochs):
            train_one_epoch(model = model, optimizer = optimizer, criterion = criterion, data_loader = train_loader, device = device)
            lr_scheduler.step()

            valid_loss, valid_acc, valid_R2, valid_mse, valid_mae, valid_spear = evaluate(model = model, data_loader = val_loader, criterion = criterion, device = device)
            print(
                'Epoch: [{}/{}], Val loss: {:.4f}, Val acc: {:.4f}, Val R2: {:.4f}, Val MSE: {:.4f}, Val MAE: {:.4f}, Val Spearman: {:.4f}'.format(epoch, n_epochs, valid_loss, valid_acc, valid_R2, valid_mse, valid_mae, valid_spear))
            
            # if valid_R2 > monitor2:
            if valid_acc > monitor1:
                # torch.save(model.state_dict(), best_weight_dir1)
                print('Now best valid corr: {:.4f}, and best weight is saved in {}.'.format(valid_acc, best_weight_dir1))
                monitor1 = valid_acc
                monitor2 = valid_R2
                monitor3 = valid_mse
                monitor4 = valid_mae
                monitor5 = valid_spear
                bad_epochs = 0
            else:
                print('No improved! Now best valid corr: {:.4f}, with corresponding R2: {:.4f}, corresponding mse: {:.4f}, corresponding mae: {:.4f}, corresponding spearman: {:.4f}.'.format(monitor1, monitor2, monitor3, monitor4, monitor5))
                bad_epochs += 1

            if bad_epochs >= patience:
                print('Early stop!')
                break

        # record the performance of the model on the task1 dataset for each fold
        Task1_acc[F] = monitor1
        Task1_R2[F] = monitor2
        Task1_mse[F] = monitor3
        Task1_mae[F] = monitor4
        Task1_spear[F] = monitor5
    
    # Print the average performance of the model on the task1 dataset for 5 folds
    print('Average corr: {:.4f}'.format(np.mean(Task1_acc)))
    print('Average R2: {:.4f}'.format(np.mean(Task1_R2)))
    print('Average mse: {:.4f}'.format(np.mean(Task1_mse)))
    print('Average mae: {:.4f}'.format(np.mean(Task1_mae)))
    print('Average spearman: {:.4f}'.format(np.mean(Task1_spear)))

if __name__ == '__main__':
    main()
