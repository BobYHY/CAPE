from model.promoter_model import PrompterModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from dataset.data_utils import get_k_fold_dataset, get_k_fold_test_dataset
from dataset.matrix import matrix_generation
from train_and_val.train_eval_utils import train_one_epoch, evaluate
from sklearn.metrics import r2_score
import sys

# set the random seed to make the experiments reproduciable
ss=3407
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
fine_tune_lr = 7e-3 
fine_tune_batch_size = 32
fine_tune_epochs = 200
train_valid_name = "Ecoli"
testname1 = "trc" 
Update_Train = False
Update_Test1 = False
K_FOLD = 5
best_weight_dir1 = './best_previous.pth'
best_weight_dir2 = './best_trc.pth'

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set the device to GPU if available
    print(device)
    CGR_N = 20      # The size of the CGR matrix

    # Generate the CGR matrix for the training and validation dataset if needed
    if Update_Train == True:
        matrix_generation(train_valid_name, CGR_N)
    if Update_Test1 == True:
        matrix_generation(testname1, CGR_N)

    # Task1_fold = get_k_fold_dataset(path_x='./data/'+train_valid_name+'_seq.npy', path_pssm='./data/'+train_valid_name+'_matrix.npy',
    #                                          path_y='./data/'+train_valid_name+'_y.npy', path_w2v='./data/word2vec.npy', Fold=K_FOLD)

    # train_dataset, val_dataset=Task1_fold[0]
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False)
    # model = PrompterModel(input_dim=in_dim,
    #                       embedding_dim=embed_dim,
    #                       depth_transformer=2,
    #                       heads_transformer=8,
    #                       dim_head_transformer=64,
    #                       attn_dropout_transformer=0.1,
    #                       ff_dropout_transformer=0.1,
    #                       dropout_CNN=0.2,
    #                       mat_size=CGR_N
    #                       )
    # model = model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.L1Loss()
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps,gamma=lr_gamma)
    # monitor1 = -1
    # monitor2 = -1
    # monitor3 = -1
    # monitor4 = -1
    # monitor5 = -1
    # bad_epochs = 0
    # for epoch in range(n_epochs):
    #     sys.stdout.flush()
    #     train_one_epoch(model=model, optimizer=optimizer, criterion=criterion, data_loader=train_loader, device=device)
    #     lr_scheduler.step()
    #     train_loss, train_acc, train_R2, _, _, _ = evaluate(model=model, data_loader=train_loader, criterion=criterion, device=device)
    #     print(
    #         'Epoch: [{}/{}], Train Loss: {:.4f}, Train acc: {:.4f}, Train R2: {:.4f}'.format(epoch, n_epochs, train_loss, train_acc, train_R2))
    #     valid_loss, valid_acc, valid_R2, valid_mse, valid_mae, valid_spear = evaluate(model=model, data_loader=val_loader, criterion=criterion, device=device)
    #     print(
    #         'Epoch: [{}/{}], Val loss: {:.4f}, Val acc: {:.4f}, Val R2: {:.4f}, Val MSE: {:.4f}, Val MAE: {:.4f}, Val Spearman: {:.4f}'.format(epoch, n_epochs, valid_loss, valid_acc, valid_R2, valid_mse, valid_mae, valid_spear))
        
    #     if valid_acc > monitor1:
    #         torch.save(model.state_dict(), best_weight_dir1)
    #         print('Now best valid corr: {:.4f}, and best weight is saved in {}.'.format(valid_acc,best_weight_dir1))
    #         monitor1 = valid_acc
    #         monitor2 = valid_R2
    #         monitor3 = valid_mse
    #         monitor4 = valid_mae
    #         monitor5 = valid_spear
    #         bad_epochs = 0
    #     else:
    #         print('No improved! Now best valid corr: {:.4f}, with corresponding R2: {:.4f}, corresponding mse: {:.4f}, corresponding mae: {:.4f}, corresponding spearman: {:.4f}.'.format(monitor1, monitor2, monitor3, monitor4, monitor5))
    #         bad_epochs += 1

    #     if bad_epochs >= patience:
    #         print('Early stop!')
    #         break

    # Get the k-fold dataset for task2 (trc dataset)
    Task2_fold = get_k_fold_test_dataset(path_x = './data/' + testname1 + '_seq.npy', path_pssm = './data/' + testname1 + '_matrix.npy',
                                             path_y = './data/' + testname1 + '_y.npy', path_w2v = './data/word2vec.npy', Fold = K_FOLD, State = 42)
    Task2_acc = [0] * K_FOLD
    Task2_R2 = [0] * K_FOLD
    Task2_mse = [0] * K_FOLD
    Task2_mae = [0] * K_FOLD
    Task2_spear = [0] * K_FOLD

    # Train the model and evaluate the performance for each fold
    for F, (fine_tune_dataset, test_dataset) in enumerate(Task2_fold):
        # Create the model
        model = PrompterModel(input_dim = 100,
                            embedding_dim = 32,
                            depth_transformer = 2,
                            heads_transformer = 8,
                            dim_head_transformer = 64,
                            attn_dropout_transformer = 0.1,
                            ff_dropout_transformer = 0.1,
                            dropout_CNN = 0.2,
                            mat_size = CGR_N
                            )
        # Load the best weight model from the task1 for subsequent fine-tuning
        best_weight = torch.load(best_weight_dir1)
        # apply the best weight to the model
        model.load_state_dict(best_weight)
        # freeze the parameters of the model
        for param in model.parameters():
            param.requires_grad = False
        
        # change the output layer of the model to the fine-tuning layer
        model.output_layer = nn.Sequential(
        nn.Linear(16 * CGR_N * CGR_N + 48 * embed_dim, 4 * CGR_N * CGR_N + 12 * embed_dim), 
        nn.ReLU(),  
        nn.Linear(4 * CGR_N * CGR_N + 12 * embed_dim, CGR_N * CGR_N + 3 * embed_dim), 
        nn.ReLU(),  
        nn.Linear(CGR_N * CGR_N + 3 * embed_dim, 128), 
        nn.ReLU(),  
        nn.Linear(128, 64), 
        nn.ReLU(),  
        nn.Linear(64, 32),
        nn.ReLU(),  
        nn.Linear(32, 8),
        nn.ReLU(),  
        nn.Linear(8, 1),
        )
        # model.output_layer = nn.Sequential(
        # nn.Linear(16*CGR_N*CGR_N, 4*CGR_N*CGR_N), 
        # nn.ReLU(),  
        # nn.Linear(4*CGR_N*CGR_N, CGR_N*CGR_N), 
        # nn.ReLU(),  
        # nn.Linear(CGR_N*CGR_N, 128), 
        # nn.ReLU(),  
        # nn.Linear(128, 64), 
        # nn.ReLU(),  
        # nn.Linear(64, 32),
        # nn.ReLU(),  
        # nn.Linear(32, 8),
        # nn.ReLU(),  
        # nn.Linear(8, 1),
        # )
        # model.output_layer = nn.Sequential(
        # nn.Linear(48*embed_dim, 12*embed_dim), 
        # nn.ReLU(),  
        # nn.Linear(12*embed_dim, 3*embed_dim), 
        # nn.ReLU(),  
        # nn.Linear(3*embed_dim, 128), 
        # nn.ReLU(),  
        # nn.Linear(128, 64), 
        # nn.ReLU(),  
        # nn.Linear(64, 32),
        # nn.ReLU(),  
        # nn.Linear(32, 8),
        # nn.ReLU(),  
        # nn.Linear(8, 1),
        # )
        model = model.to(device)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.output_layer.parameters(), lr = fine_tune_lr)

        print(f'Fold {F+1}')

        # Create the data loader for fine-tuning and testing
        fine_tune_loader = DataLoader(dataset = fine_tune_dataset, batch_size = fine_tune_batch_size, shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = len(test_dataset), shuffle = False)
        fine_tune_monitor1 = -1
        fine_tune_monitor2 = -1
        fine_tune_monitor3 = -1
        fine_tune_monitor4 = -1
        fine_tune_monitor5 = -1

        # Fine-tune the model
        for epoch in range(fine_tune_epochs):
            sys.stdout.flush()
            train_one_epoch(model = model, optimizer = optimizer, criterion = criterion, data_loader = fine_tune_loader, device = device)
            test_loss, test_acc, test_R2, test_mse, test_mae, test_spear = evaluate(model = model, data_loader = test_loader, criterion = criterion, device = device)
            print('Fine_tune_Epoch: [{}/{}], Test loss: {:.4f}, Test acc: {:.4f}, Test R2: {:.4f}, Test MSE: {:.4f}, Test MAE: {:.4f}, Test spearman: {:.4f}'.format(epoch, fine_tune_epochs, test_loss, test_acc, test_R2, test_mse, test_mae, test_spear))
            
            if test_R2 > fine_tune_monitor1:
                fine_tune_monitor1 = test_R2
                fine_tune_monitor2 = test_acc
                fine_tune_monitor3 = test_mse
                fine_tune_monitor4 = test_mae
                fine_tune_monitor5 = test_spear
                # torch.save(model.state_dict(), best_weight_dir2)
            else:
                print('No improved! Now best R2: {:.4f}, with corresponding corr: {:.4f}, corresponding mse: {:.4f}, corresponding mae: {:.4f}, corresponding spearman: {:.4f}.'.format(fine_tune_monitor1, fine_tune_monitor2, fine_tune_monitor3, fine_tune_monitor4, fine_tune_monitor5))
        
        # record the performance of the model on the task2 dataset for each fold
        Task2_R2[F] = fine_tune_monitor1
        Task2_acc[F] = fine_tune_monitor2
        Task2_mse[F] = fine_tune_monitor3
        Task2_mae[F] = fine_tune_monitor4
        Task2_spear[F] = fine_tune_monitor5

    # Print the average performance of the model on the task2 dataset for 5 folds
    print('Average R2: {:.4f}'.format(np.mean(Task2_R2)))
    print('Average corr: {:.4f}'.format(np.mean(Task2_acc)))
    print('Average mse: {:.4f}'.format(np.mean(Task2_mse)))
    print('Average mae: {:.4f}'.format(np.mean(Task2_mae)))
    print('Average spearman: {:.4f}'.format(np.mean(Task2_spear)))

if __name__ == '__main__':
    main()
