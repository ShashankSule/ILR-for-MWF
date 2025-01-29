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
from scipy.optimize import curve_fit
import functools
from scipy import special
from multiprocessing import Pool, freeze_support
import multiprocess as mp

if __name__ == "__main__":
    curr_path = os.path.abspath('')

    for snr in [5.0,50.0,100.0]:
        print(f"=========================SNR: {snr}===================================")
        for mod_type in ["NDReg", "NDND"]:
            dS_Dict = {"Training" : None, "Validation": None}
            # mm = MinMaxScaler()
            means = None
            stdevs = None

            for dp in ["Training", "Validation"]:
                estim_path = os.path.relpath(f"../Lambda_TrainingData/3PE_ReconSignals/REG__DenseGraph_ExpectationRicianNoise_1000NR_SNR_{snr}_{dp}Data.feather", curr_path)
            #     perturbed_path = os.path.relpath(f"../Lambda_TrainingData/3PE_ReconSignals/PerturbedNN_3PE_BrainDataSignals_Myelin_32Pts_SNR_900.0_{dp}Data.feather", curr_path)
                # MDN = 3PE_MDNSignals_Myelin_128Pts_SNR_900.0_{dp}Data.feather", curr_path)
                # Oracle = 3PE_OracleSignals_Myelin_128Pts_SNR_900.0_{dp}Data.feather", curr_path)
                # OG = 3PE_ReconSignals_LambdaNN_Myelin_128Pts_SNR_900.0_{dp}Data.feather", curr_path)
                PE_path = os.path.relpath(f"../Lambda_TrainingData/LambdaGeneration/DenseGraph_ExpectationRicianNoise_1000NR_SNR_{snr}_{dp}Data.feather", curr_path)
                means, stdevs = None, None
                if dp !="Training":
                    means = torch.tensor([0])
                    stdevs = torch.tensor([1])

                dS_Dict[dp] = toDataset(dp, estim_path, PE_path, means = means, stdevs = stdevs)
            
            
            if mod_type == "NDND":
                dS_Dict["Training"].training_tensor_proc = dS_Dict["Training"].training_tensor_proc[:,0,:].repeat(1,2)
                dS_Dict["Validation"].training_tensor_proc = dS_Dict["Validation"].training_tensor_proc[:,0,:].repeat(1,2)

            else:
                dS_Dict["Training"].training_tensor_proc = dS_Dict["Training"].training_tensor_proc.flatten(1)
                dS_Dict["Validation"].training_tensor_proc = dS_Dict["Validation"].training_tensor_proc.flatten(1)

            
        #     dS_Dict["Training"].training_tensor_proc[:,:2] = dS_Dict["Training"].training_tensor_proc[:,:2].reciprocal()
        #     dS_Dict["Validation"].training_tensor_proc[:,:2] = dS_Dict["Validation"].training_tensor_proc[:,:2].reciprocal()
        #     dS_Dict["Testing"].training_tensor_proc[:,:2] = dS_Dict["Testing"].training_tensor_proc[:,:2].reciprocal()
            
            Z = dict()
            Z['num_epoch'] = 45 #Afkham et al.
            Z['lr'] = 0.0001
            Z['batch_size'] = 64 #(64 in Afkham)
            Z["batch_size_validation"] = 500
            Z['N_train'] = len(dS_Dict["Training"])
            Z['N_validation'] = len(dS_Dict["Validation"])
            
            seed = 748
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
                        
                
            device = torch.device("cuda:6") #WAS CUDA:7

            
            training_loader_4 = DataLoader(dS_Dict["Training"], shuffle = True, batch_size= Z['batch_size'])
            validation_loader_4 = DataLoader(dS_Dict["Validation"], batch_size= Z['batch_size_validation'])

            train_bool = True
            if train_bool:
                input_size = 64
                H1,H2,H3,H4= 32,256,256,32
                output_size = 3
                model_orig = LambdaTraining_FC_NN(input_size, H1, H2, H3, H4, output_size)
                model_orig = model_orig.to(device)
                optimizer = torch.optim.Adam(model_orig.parameters(), lr= Z['lr']) #Mirage: Changed to AdamW --> implement LM like Afkham??, Using default Beta1 and Beta2 params for momentum change; set regularization term (if any) to 0?
                
                scaling = 100.0
                scale_factor = torch.tensor([1.0,1.0,scaling], dtype = float, device = device)
                l_p = 2
                def L_loss(pred, actual):
                    return (((pred-actual)*scale_factor)**l_p).mean()
                myLoss = L_loss

                if mod_type == "NDND":
                    DATA_NAME = f"3P_RicianNoise__ILR__NDND__SNR_{snr}"
                    
                else:    
                    DATA_NAME = f"3P_{mod_type}__ILR__NDReg__SNR_{snr}"
                
                NET_NAME_2 = DATA_NAME + f'L{l_p}Loss_TE32_B{Z["batch_size"]}_lr1e4'

                NET_DIR_2 = "results/_snr" + str(snr)
                NET_PATH_2 = NET_DIR_2 + '/' + NET_NAME_2

                best_vloss = np.inf

                best_epoch = 0
                training_loss_e, validation_loss_e = torch.zeros(Z['num_epoch']), torch.zeros(Z['num_epoch'])

                steps_per_epoch = len(training_loader_4)

                for epoch in range(Z['num_epoch']):
                    model_orig.train()
                    running_loss = 0.0
                    with tqdm(training_loader_4, unit = "batch") as tepoch:

                        for batch_idx, (noisy_decay, targets) in enumerate(tepoch):

                            tepoch.set_description(f"Epoch {epoch +1}")

                            curr_lr = optimizer.param_groups[0]['lr']
                            noisy_decay, targets = noisy_decay.to(device), targets.to(device)

                            optimizer.zero_grad()

                            predictions = model_orig(noisy_decay)
                            loss = myLoss(predictions, targets)

                            loss.backward()

                            optimizer.step()
                            single_loss = loss.item()
                            running_loss += single_loss*noisy_decay.size(0) 
                            tepoch.set_postfix(loss_Example = single_loss, lr = curr_lr)
                        training_loss_e[epoch] = running_loss/Z['N_train']

                    # Validation===============================================================================
                    with torch.no_grad():
                        model_orig.eval()
                        valid_loss = 0.0
                        with tqdm(validation_loader_4, unit = "batch") as tepoch_valid:
                            for batch_idx_val, (noisy_decay, targets) in enumerate(tepoch_valid):
                                tepoch_valid.set_description(f"Validation Epoch {epoch +1}")

                                noisy_decay, targets = noisy_decay.to(device), targets.to(device)
                                predictions = model_orig(noisy_decay)
                                loss = myLoss(predictions, targets)
                                valid_loss += loss.item()*noisy_decay.size(0)
                            validation_loss_e[epoch] = valid_loss / Z["N_validation"]  

                        print(f'Epoch {epoch+1}\nTraining Loss: {training_loss_e[epoch]}\nValidation Loss: {validation_loss_e[epoch]}')
                        saved = validation_loss_e[epoch] < best_vloss
                        if saved:
                            best_vloss = validation_loss_e[epoch]
                            best_epoch = epoch + 1
                            model_path = NET_PATH_2 + ".pth"
                            torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': model_orig.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    }, model_path)
                        print("Best vLoss:", best_vloss,  "\tBest Epoch: ", best_epoch,"\tSaved: ", saved)
                torch.save(training_loss_e, f"{NET_DIR_2}/TRAININGLOSS_{NET_NAME_2}.pt")
                torch.save(validation_loss_e, f"{NET_DIR_2}/VALIDATIONLOSS_{NET_NAME_2}.pt")