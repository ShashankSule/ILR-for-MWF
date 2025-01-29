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

GENERATE = True
if GENERATE:
    if __name__ == "__main__":
        curr_path = os.path.abspath('')
        
        # p = 1
        # def MPE(pred,actual):
        #     return ((torch.abs(pred-actual))**p).mean()

        for snr in [5.0,50.0,100.0]:
            
            print(f"Beginning SNR {snr}...")
            convolutional = True
            
            og_train_path = os.path.relpath(f"../Lambda_TrainingData/LambdaTraining/DenseGraph_ExpectationRicianNoise_1000NR_SNR_{snr}_TrainingData.feather",curr_path)
            training_dataset = initDataset(og_train_path, set_type = "training", type1 = "standardized", convolutional = convolutional)
            


            training_path_2 = os.path.relpath(f"../Lambda_TrainingData/LambdaGeneration/DenseGraph_ExpectationRicianNoise_1000NR_SNR_{snr}_TrainingData.feather",curr_path)
            testing_path_2 = os.path.relpath(f"../Lambda_TrainingData/LambdaGeneration/DenseGraph_ExpectationRicianNoise_1000NR_SNR_{snr}_TestingData.feather", curr_path)

            training_dataset_2 = initDataset(training_path_2, set_type = "validation", select_target = "Lambda", type1 = "standardized", mean_targ= training_dataset.mean2, std_targ = training_dataset.stdev, convolutional = convolutional)
            # validation_dataset_2 = initDataset(validation_path_2, set_type = "validation", select_target = "Lambda", type1 = "standardized", mean_targ= training_dataset.mean2, std_targ = training_dataset.stdev, convolutional = convolutional)
            testing_dataset_2 = initDataset(testing_path_2, set_type = "validation", select_target = "Lambda", type1 = "standardized", mean_targ= training_dataset.mean2, std_targ = training_dataset.stdev, convolutional = convolutional)

            training_loader_2 = DataLoader(training_dataset_2, shuffle = False, batch_size= 1500)
            # validation_loader_2 = DataLoader(validation_dataset_2, shuffle = False, batch_size= 1500)
            testing_loader_2 = DataLoader(testing_dataset_2, shuffle = False, batch_size= 1500)

            device = torch.device("cuda:6") #WAS CUDA:7
            
            DATA_NAME = f"Experimentation_DenseRician_LambdaNN_SNR_{snr}"
            NET_NAME = DATA_NAME + f'MPE_{1}_lr1e3_B{512}'

            NET_DIR = "results/_snr" + str(snr)
            NET_PATH = NET_DIR + '/' + NET_NAME
            trained_model_path = NET_PATH + ".pth"

            compiled_model = LambdaTraining_FC_NN_Convolutional_SELU(in_channel=1, out1 = 128, out2 = 256, k1 = 5, k2 = 3, fc1 = 512, fc2 = 64, out_dim=1)
            compiled_model = compiled_model.to(device)
            compiled_model.load_state_dict(torch.load(trained_model_path)['model_state_dict'])
            compiled_model.eval()

            purposeList = ["Testing"]
            for i, dataload in enumerate([testing_loader_2]):
                with torch.no_grad():
                    training_preds = torch.empty(len(dataload.dataset))
                    curr_pos = 0
                    end_pos = 0
                    for idx, (noisy_decay, targets) in enumerate(tqdm(dataload, unit = "batch")):
                        batch_size = noisy_decay.size(0)
                        end_pos += batch_size
                        assert(torch.all(noisy_decay == (dataload.dataset.training_tensor_proc)[curr_pos : end_pos]))
                        noisy_decay = noisy_decay.to(device)
            #             noisy_decay = noisy_decay.to(device)
                        predictions = compiled_model(noisy_decay.float())          
                        # loss = mdn_loss_fn(pi_variable, sigma_variable, mu_variable, targets.float())

                        # rL += loss.item() * batch_size
                        training_preds[curr_pos : end_pos] = (predictions.squeeze(1) * training_dataset.stdev) + training_dataset.mean2
                        # predictions = model_testing(noisy_decay.float())
                        # training_preds[idx*batch_size : batch_size*(idx+1)] = predictions.squeeze(1)
                        curr_pos += batch_size
            #         print(training_preds.shape)
                    assert(not training_preds.isnan().any())
                    assert(not training_preds.isinf().any())
                    torch.save(training_preds, os.path.relpath(f"../Lambda_TrainingData/LambdaGeneration/LAMBDAS__DenseGraph_ExpectationRicianNoise_1000NR_SNR_{snr}_{purposeList[i]}Data.pt", curr_path))            



    def mycurvefit_l2Regularized_3param(i, datatype = None, signalType="biexponential", lb_T21=0.0, lb_T22=0.0, lb_c1=0.0, ub_T21=np.inf, ub_T22=np.inf, ub_c1=np.inf): #c1 ub = 1.0??
            
        dataPurpose = datatype
        SNR = 100.0
        times = np.linspace(8, 256, 32)

        D = np.array([1.0,  # T2,1

                    1.0,  # T2,2

                    100.0])  # C1
        ld = data[i][0]
        p_0 = data[i][3:6]

        d = data[i][6:]
        assert(len(d) == 32)

        def signal(t_vec, p1, p2, p3):
        #         p1 = t1
        #         p2 = t2
        #         p3 = c1
            if signalType == "biexponential":
                return p3*np.exp(-t_vec/p1) + (1.0-p3)*np.exp(-t_vec/p2)

        def expectation_Rice(xdata,t21,t22, c1):
            sigma=np.max(d)/SNR
            t_vec = xdata[:len(d)]
            ld_val = xdata[len(d)]
            alpha=(signal(t_vec, t21, t22, c1)/(2*sigma))**2
            Expectation = sigma*np.sqrt(np.pi/2)*((1+2*alpha)*special.ive(0, alpha) + 2*alpha*special.ive(1,alpha))
            params = np.array([t21, t22, c1], dtype=np.float64)
            penalty_vec = ld_val*np.multiply(D,params)
            return np.concatenate((Expectation, penalty_vec))                              



        # B) Bounds on the parameters

        #lb_T21, ub_T21 = 0.0, np.inf  # lb can be small & positive to enforce nonnegativity

        #lb_T22, ub_T22 = 0.0, np.inf



        # C) Fit given dependent variable to curve and independent variable

        # Uses 2 point finite-differences to approximate the gradient.

        # Could also use 3 point or take exact gradient.

        t_dim = times.ndim 

        indep_var = np.concatenate((times, #|| Time lambda|| - Pg 95 Aster 3rd Ed.

                                    np.array(ld,ndmin=t_dim)))



        d_dim = d.ndim

        depen_var = np.concatenate((d, np.array(0.0, ndmin=d_dim), np.array(0.0,ndmin=d_dim), np.array(0.0, ndmin=d_dim))) #||ND 0 0 0||


        try:
        #         print("Initial Guess: ", p_0)
            opt_val = curve_fit(expectation_Rice, indep_var, depen_var,  # curve, xdata, ydata

                                p0=p_0,  # initial guess

                                bounds=([lb_T21, lb_T22, lb_c1], [ub_T21, ub_T22, ub_c1]),

                                method="trf",

                                max_nfev=1000)
            #print('!!!!!!!!!!', opt_val)
        except RuntimeError:
            opt_val = (np.asarray([66.08067015, 66.44472936, 45.5891436 ]), np.asarray([[ 6.51065099e+03, -6.53878183e+03,  1.61616729e+06], [-6.53878183e+03,  6.85069747e+03, -1.65784453e+06], [ 1.61616729e+06, -1.65784453e+06,  4.05431645e+08]]))

            print("maximum number of function evaluations == exceeded")





        # returns estimate. second index in estimated covariance matrix

    





            # returns estimate. second index in estimated covariance matrix

        T21_ld, T22_ld, c1_ret = opt_val[0]
    # Enforces T21 <= T22
        if T21_ld.size == 1:
            T21_ld = T21_ld.item()
            T22_ld = T22_ld.item()
            c1_ret = c1_ret.item()

        if T21_ld > T22_ld:
            T21_ld_new = T22_ld
            T22_ld = T21_ld
            T21_ld = T21_ld_new
            c1_ret = 1.0 - c1_ret
            assert (T21_ld != T22_ld)


        # return opt_val[0]
        if i == None:
            return d, ld, c1_ret, T21_ld, T22_ld 
        else:
            return d, ld, c1_ret, T21_ld, T22_ld, i


    print("===================STARTING RegTraj MP==========================================")
    for snr in [5.0,50.0,100.0]:
        for dp in ["Testing"]:

            curr_path = os.path.abspath('')

            convolutional = True
            training_path_3 = os.path.relpath(f"../Lambda_TrainingData/LambdaGeneration/DenseGraph_ExpectationRicianNoise_1000NR_SNR_{snr}_{dp}Data.feather",curr_path)
            training_dataset_3 = initDataset(training_path_3, set_type = "validation", select_target = ["T21_t", "T22_t", "c1_t"], type1 = "standardized", mean_targ= torch.tensor([0]), std_targ = torch.tensor([1]), convolutional = convolutional)


            noisy_signals = (training_dataset_3.training_tensor_proc).squeeze(1)
            lambdas_t2s = 10**torch.load(os.path.relpath(f"../Lambda_TrainingData/LambdaGeneration/LAMBDAS__DenseGraph_ExpectationRicianNoise_1000NR_SNR_{snr}_{dp}Data.pt", curr_path))

            c1_t2s = training_dataset_3.master_frame["c1_t"]
            c2_t2s = 1.0 - training_dataset_3.master_frame["c1_t"]
            p_s = training_dataset_3.target_tensor_proc


            # mpp.set_start_method('spawn', force=  True) #Artifact -- may not be needed anymore
            print("LAMBDA NANs: ", np.where(np.isnan(lambdas_t2s)))
            print("LAMBDA Infs: ", np.where(np.isinf(lambdas_t2s)))
            print("ND NANs: ", np.where(np.isnan(noisy_signals)))
            print("ND Infs: ", np.where(np.isinf(noisy_signals)))

            j_NDs = torch.cat((lambdas_t2s.view(-1,1),
                                torch.tensor(c1_t2s.values).view(-1,1), torch.tensor(c2_t2s.values).view(-1,1),
                                p_s,
                                noisy_signals), dim = 1)

            print("Starting MP...")
            data = j_NDs.detach().numpy()

            if __name__ == '__main__':
                freeze_support()
                print("Finished Assignments...")
                num_cpus_avail = 64
                print("Using Super Computer")

                print(f"Building {dp} Dataset...")
                lis = []

                with mp.Pool(processes = num_cpus_avail) as pool:

                    with tqdm(total=data.shape[0]) as pbar:
                        for ND, ld, c1, t21, t22, k in pool.imap_unordered(
                            functools.partial(mycurvefit_l2Regularized_3param, datatype=dp), range(data.shape[0]), chunksize = 250):
                            # if k == 0:
                                # print("Starting...")

                            lis.append(np.concatenate([np.array([c1, t21, t22, k, ld]), ND], dtype=np.float64))

                            pbar.update()
                #             # break
                    pool.close() #figure out how to implement
                    pool.join()
                T2_ld = np.stack([row for row in lis], axis = 0)
                print(T2_ld.shape)
                df = pd.DataFrame(index = range(T2_ld.shape[0]), columns = ["c1_ld","t21_ld", "t22_ld", "Index", "lambda", "ND"])
                df[["c1_ld", "t21_ld", "t22_ld", "Index", "lambda"]] = T2_ld[:,:-32]
                df["ND"] = [T2_ld[i,-32:] for i in range(T2_ld.shape[0])]
                print("DATAFRAME: ", df.shape)
                df.to_feather(f"../Lambda_TrainingData/3PE_ReconSignals/REG__DenseGraph_ExpectationRicianNoise_1000NR_SNR_{snr}_{dp}Data.feather")