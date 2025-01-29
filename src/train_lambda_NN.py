#Libraries
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt

import numpy as np

import os

import torch

import torch.nn as nn

import random

from torchvision import models

from torch.utils.data import DataLoader
#Access all files within make dataset
import sys
parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)


#Network Architecture File
from FC_NN_Architecture import *
# Dataset Loader File
from DF_DataLoader import initDataset, toDataset

# Data visualization of NN
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange

#Testing Files
from makeSignals import myTrueModel
from regTraj import make_l2_trajectory
import pickle
from torchinfo import summary



if __name__ == "__main__":
    curr_path = os.path.abspath('')
    
    p = 1
    def MPE(pred,actual):
        return ((torch.abs(pred-actual))**p).mean()

    for snr in [5.0,50.0,100.0]:
        print(f"Beginning SNR {snr}...")
        training_path = os.path.relpath(f"../Lambda_TrainingData/LambdaTraining/DenseGraph_ExpectationRicianNoise_1000NR_SNR_{snr}_TrainingData.feather",curr_path)
        validation_path = os.path.relpath(f"../Lambda_TrainingData/LambdaTraining/DenseGraph_ExpectationRicianNoise_1000NR_SNR_{snr}_ValidationData.feather", curr_path)

        convolutional = True
        training_dataset = initDataset(training_path, set_type = "training", type1 = "standardized", convolutional = convolutional)
        validation_dataset = initDataset(validation_path, set_type = "validation", type1 = "standardized", mean_targ= training_dataset.mean2, std_targ= training_dataset.stdev, convolutional = convolutional)
        
        print("MEAN: ",training_dataset.mean2)
        print("STDev: ",training_dataset.stdev)
        device = torch.device("cuda:6") #WAS CUDA:7

        P = dict()
        P['num_epoch'] = 45
        P['lr'] = 0.0001
        # P['lr'] = np.logspace(-5,1,1000)
        P['batch_size'] = 512 #(64 in Afkham)
        P['batch_size_validation'] = 1048 #(64 in Afkham)
        P['N_train'] = len(training_dataset)
        P['N_validation'] = len(validation_dataset)

        seed = 657
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        compiled_model = LambdaTraining_FC_NN_Convolutional_SELU(in_channel=1, out1 = 128, out2 = 256, k1 = 5, k2 = 3, fc1 = 512, fc2 = 64, out_dim=1)
        compiled_model = compiled_model.to(device)
        optimizer = torch.optim.Adam(compiled_model.parameters(), lr = P['lr'])

        summary(compiled_model, (P["batch_size"],1,32))

        training_loader = DataLoader(training_dataset, shuffle = True, batch_size= P['batch_size'], pin_memory = True, persistent_workers=True, num_workers=32)
        validation_loader = DataLoader(validation_dataset, batch_size= P['batch_size_validation'])

        myLoss = MPE

        DATA_NAME = f"Experimentation_DenseRician_LambdaNN_SNR_{snr}"
        NET_NAME = DATA_NAME + f'MPE_{p}_lr1e3_B{P["batch_size"]}'

        NET_DIR = "results/_snr" + str(snr)
        NET_PATH = NET_DIR + '/' + NET_NAME

        if not os.path.exists(NET_DIR): os.makedirs(NET_DIR)


        best_vloss = np.inf
        best_epoch = 1
        training_loss_e, validation_loss_e = torch.zeros(P['num_epoch']), torch.zeros(P['num_epoch'])

        for epoch in range(P['num_epoch']):

            #======================================= Training ==============================================================================
            
            compiled_model.train()
            running_loss = 0.0
            with tqdm(training_loader, unit = "batch", position=0) as tepoch:
                
                for batch_idx, (noisy_decay, targets) in enumerate(tepoch):
                    
                    tepoch.set_description(f"Epoch {epoch + 1}")
                    curr_lr = optimizer.param_groups[0]['lr']

                    noisy_decay, targets = noisy_decay.to(device), targets.unsqueeze(1).to(device)
                    optimizer.zero_grad()
                    predictions = compiled_model(noisy_decay.float())
                    loss = myLoss(predictions.float(), targets.float())
                    loss.backward()
                    optimizer.step()
                    single_loss = loss.item()
                    running_loss += single_loss*noisy_decay.size(0) 
                    
                    tepoch.set_postfix(loss_Example = single_loss, lr = curr_lr)

                training_loss_e[epoch] = running_loss/P['N_train']

            #======================================= Validation ===============================================================================

            with tqdm(validation_loader, unit = "batch", position=1) as vpoch:

                with torch.no_grad():
                    compiled_model.eval()
                    valid_loss = 0.0
                    for batch_idx_val, (noisy_decay, targets) in enumerate(vpoch): #Will only loop once since not batching
                        noisy_decay, targets = noisy_decay.to(device), targets.unsqueeze(1).to(device)
                        predictions = compiled_model(noisy_decay.float())
                        loss = myLoss(predictions.float(), targets.float())
                        valid_loss += loss.item()*noisy_decay.size(0)
                    validation_loss_e[epoch] = valid_loss / P["N_validation"]  
                    
                    print(f'Epoch {epoch+1}\nTraining Loss: {training_loss_e[epoch]}\nValidation Loss: {validation_loss_e[epoch]}')
                    saved = validation_loss_e[epoch] < best_vloss
                    if saved:
                        best_vloss = validation_loss_e[epoch]
                        best_epoch = epoch + 1
                        model_path = NET_PATH + ".pth"
                        # f"__Epoch{epoch+1}.pth"
                        torch.save({
                                'epoch': epoch,
                                'model_state_dict': compiled_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, model_path)
                    print("Best vLoss:", best_vloss,  "\tBest Epoch: ", best_epoch,"\tSaved: ", saved)
        torch.save(training_loss_e, f"{NET_DIR}/TRAININGLOSS_{NET_NAME}.pt")
        torch.save(validation_loss_e, f"{NET_DIR}/VALIDATIONLOSS_{NET_NAME}.pt")

