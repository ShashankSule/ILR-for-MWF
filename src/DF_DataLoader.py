from ..makeSignals import myTrueModel, myTrueModel_2param

import matplotlib.pyplot as plt

import numpy as np

import torch

import random

from torch.utils.data import Dataset

from typing import List

import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Lambda_FULL(Dataset):
    
    def __init__(self, pkl_file):
        print('BEGIN MULTIPLE DECAYS!!!')
        """

        Input:

        ------

        1. csv_file (string): Path to the csv file containing the data set. This

            is a saved DataFrame.

        2. target_names (list of strings): The names of the target values the NN

            aims to predict. These are keys in the DataFrame saved to the csv

            file that csv_file points to.

        3. time_bounds (list of 2 floats): Bounds on the times the decay is

            measured at.  The lower bound is time_bounds[0] and the upper bound

            is time_bounds[1].

        4. decay_input (List of strings): When making an instance of the

        Multiple Decay DataSet, the names of the decay input

        to be included. For example, 'A', 'B', or 'ND'.

        """


#         file = open(pkl_file, 'rb')
        self.master_frame = pd.read_feather(pkl_file)
#         file.close()
        
#         rand = np.random.randint(len(self.master_frame), size=2) #REMOVE LATER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.training_frame = self.master_frame["Noisy Signal"].apply(torch.from_numpy)
        self.target_frame = self.master_frame["Lambda"].apply(torch.tensor, dtype=float) #Train on exponents, or 10**exp??

        self.num_times = len(self.training_frame[0]) #alter to match noisy signal only
        

    def __len__(self):
        #decay_input = ['ND', 'NB']
        """

        Returns the number of samples in the data set

        """

        return len(self.training_frame)

    def __getitem__(self, idx):

        """

        Returns samples at idx.

        - If idx is an int, the output is a pair of 1-d Tensors: the first is

        the regularized decays stacked horizontally and the second is the

        corresponding target parameters.

        - If idx is a list, it returns two Tensors, each with 2 dimensions.

        The first index points to the particular sample and the second index

        points to an intensity at some time for some regularized decay.

        """
        #decay_input = ['ND', 'NB']
        ND_tensor = self.training_frame[idx]
        target = self.target_frame[idx]

        # print(multiple_decays)
        return ND_tensor, target



class Lambda(Dataset):
    
    def __init__(self, pkl_file):
        print('BEGIN MULTIPLE DECAYS!!!')
        """

        Input:

        ------

        1. csv_file (string): Path to the csv file containing the data set. This

            is a saved DataFrame.

        2. target_names (list of strings): The names of the target values the NN

            aims to predict. These are keys in the DataFrame saved to the csv

            file that csv_file points to.

        3. time_bounds (list of 2 floats): Bounds on the times the decay is

            measured at.  The lower bound is time_bounds[0] and the upper bound

            is time_bounds[1].

        4. decay_input (List of strings): When making an instance of the

        Multiple Decay DataSet, the names of the decay input

        to be included. For example, 'A', 'B', or 'ND'.

        """


        file = open(pkl_file, 'rb')
        self.master_frame = pickle.load(file)
        file.close()
        
#         rand = np.random.randint(len(self.master_frame), size=2) #REMOVE LATER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.training_frame = self.master_frame["Noisy Signal"][:1000]
        self.target_frame = self.master_frame["Lambda"][:1000] #Train on exponents, or 10**exp??

        self.num_times = len(self.training_frame[0]) #alter to match noisy signal only
        

    def __len__(self):
        #decay_input = ['ND', 'NB']
        """

        Returns the number of samples in the data set

        """

        return len(self.training_frame)

    def __getitem__(self, idx):

        """

        Returns samples at idx.

        - If idx is an int, the output is a pair of 1-d Tensors: the first is

        the regularized decays stacked horizontally and the second is the

        corresponding target parameters.

        - If idx is a list, it returns two Tensors, each with 2 dimensions.

        The first index points to the particular sample and the second index

        points to an intensity at some time for some regularized decay.

        """
        #decay_input = ['ND', 'NB']
        ND_tensor = self.training_frame[idx]
        target = self.target_frame[idx]

        # print(multiple_decays)
        return ND_tensor, target


    
class Lambda_Standardized(Dataset):
    
    def __init__(self, pkl_file, set_type, select_target, mean1 = None, std1 = None, mean_targ1 = None, std_targ1 = None, convolutional = False):
        print('BEGIN MULTIPLE DECAYS!!!')
        """

        Input:

        ------

        1. csv_file (string): Path to the csv file containing the data set. This

            is a saved DataFrame.

        2. target_names (list of strings): The names of the target values the NN

            aims to predict. These are keys in the DataFrame saved to the csv

            file that csv_file points to.

        3. time_bounds (list of 2 floats): Bounds on the times the decay is

            measured at.  The lower bound is time_bounds[0] and the upper bound

            is time_bounds[1].

        4. decay_input (List of strings): When making an instance of the

        Multiple Decay DataSet, the names of the decay input

        to be included. For example, 'A', 'B', or 'ND'.

        """


        self.master_frame = pd.read_feather(pkl_file)
        
#         self.master_frame.group_by(["T21_t", "T22_t", "c1_t"])
        
#         training_frame = 
        training_tensor_preproc = torch.from_numpy(np.stack(self.master_frame["Noisy Signal"]))

#         torch.stack([row for row in training_frame.values])
        if select_target != "Reg":
            target_frame = self.master_frame[select_target] #Train on exponents, or 10**exp??

        # reg_sig = myTrueModel(times.reshape(1,-1), reg_params[:,2].reshape(-1,1), reg

        if select_target == "Lambda":
            
            target_tensor_preproc = torch.log10(torch.tensor(target_frame.values))

            
        elif select_target == "Reg":
            reg_params = torch.from_numpy(self.master_frame[["T21_est", "T22_est", "c1_est"]].values)
            times = np.linspace(11.3, 225.0*3, 32)
            reg_sig = myTrueModel(times.reshape(1,-1), reg_params[:,2].reshape(-1,1), reg_params[:,0].reshape(-1,1), reg_params[:,1].reshape(-1,1))
            target_tensor_preproc = reg_sig

        elif select_target == "TrueSignal":
#             Unsqueeze Target!!! and recalc mean subtraction
            target_tensor_preproc = torch.stack([row for row in target_frame.values])
        elif select_target == ["T21_t", "T22_t", "c1_t"]:
            t21_tensor = torch.tensor(self.master_frame["T21_t"])
            t22_tensor = torch.tensor(self.master_frame["T22_t"])
            c1_tensor = torch.tensor(self.master_frame["c1_t"])
            
            target_tensor_preproc = torch.stack([t21_tensor, t22_tensor, c1_tensor], dim =1)
        
        self.target_tensor_proc = target_tensor_preproc.float()
        
        
        
        
        if convolutional:
            self.training_tensor_proc = training_tensor_preproc.unsqueeze(1).float()
#             self.target_tensor_proc = self.target_tensor_proc.unsqueeze(1)
            
            if set_type == "training":
                # self.mean_train, self.std_train = torch.tensor([0]), torch.tensor([1.0])
                # self.training_tensor_proc = (self.training_tensor_proc - self.mean_train[None,:,None])/self.std_train[None,:,None]

                self.mean2, self.stdev = self.target_tensor_proc.mean(), self.target_tensor_proc.std()
                self.target_tensor_proc = (self.target_tensor_proc - self.mean2)/(self.stdev).float()

#                 self.training_tensor_proc.mean([0,2]), self.training_tensor_proc.std([0,2])
            else:
                self.target_tensor_proc = (self.target_tensor_proc - mean_targ1)/(std_targ1).float()
#                 self.training_tensor_proc = (self.training_tensor_proc - mean1[None,:,None])/(std1[None,:, None])
                
        
        elif not convolutional:
            self.target_tensor_proc = self.target_tensor_proc.float()
        else:
            raise NameError("Invalid Convolutional")
         
        
        
#         self.training_tensor_proc = torch.cat([self.training_tensor_proc, self.training_tensor_proc], dim = 1)
        
#         rand = np.random.randint(len(self.master_frame), size=2) #REMOVE LATER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        
        

        self.num_times = len(self.training_tensor_proc[0]) #alter to match noisy signal only
        

    def __len__(self):
        #decay_input = ['ND', 'NB']
        """

        Returns the number of samples in the data set

        """

        return len(self.training_tensor_proc)

    def __getitem__(self, idx):

        """

        Returns samples at idx.

        - If idx is an int, the output is a pair of 1-d Tensors: the first is

        the regularized decays stacked horizontally and the second is the

        corresponding target parameters.

        - If idx is a list, it returns two Tensors, each with 2 dimensions.

        The first index points to the particular sample and the second index

        points to an intensity at some time for some regularized decay.

        """
        #decay_input = ['ND', 'NB']
        ND_tensor = self.training_tensor_proc[idx]
        target = self.target_tensor_proc[idx]

        # print(multiple_decays)
        return ND_tensor, target
    

class Lambda_trueSigGen(Dataset):
    def __init__(self, pkl_file, set_type, select_target, trueSig_Data, mean1 = None, std1 = None, convolutional = False):
        print('BEGIN MULTIPLE DECAYS!!!')
        
        file = open(pkl_file, 'rb')
        self.master_frame = pickle.load(file)
        file.close()
        
        training_frame = self.master_frame["Noisy Signal"]
        training_tensor_preproc = torch.stack([row for row in training_frame.values])

        reg_params = torch.from_numpy(self.master_frame[["T21_est", "T22_est", "c1_est"]].values)

        times = np.linspace(11.3, 225.0*3, 32)
        reg_sig = myTrueModel(times.reshape(1,-1), reg_params[:,2].reshape(-1,1), reg_params[:,0].reshape(-1,1), reg_params[:,1].reshape(-1,1))
        
        if convolutional:
            self.training_tensor_proc = torch.cat([training_tensor_preproc.unsqueeze(1), trueSig_Data.unsqueeze(1)], dim = 1)
            
            if set_type == "training":
                self.mean2, self.stdev = self.training_tensor_proc.mean([0,2]), self.training_tensor_proc.std([0,2])
                self.training_tensor_proc = (self.training_tensor_proc - self.mean2[None,:,None])/(self.stdev[None,:,None])
            else:
                self.training_tensor_proc = (self.training_tensor_proc - mean1[None,:,None])/(std1[None,:,None])
                
            t21_tensor = torch.tensor(self.master_frame["T21_t"])
            t22_tensor = torch.tensor(self.master_frame["T22_t"])
            self.target_tensor_proc = torch.stack([t21_tensor, t22_tensor], dim =1)
        
    
    
    def __len__(self):
        #decay_input = ['ND', 'NB']
        """

        Returns the number of samples in the data set

        """

        return self.training_tensor_proc.shape[0]

    def __getitem__(self, idx):

        """

        Returns samples at idx.

        - If idx is an int, the output is a pair of 1-d Tensors: the first is

        the regularized decays stacked horizontally and the second is the

        corresponding target parameters.

        - If idx is a list, it returns two Tensors, each with 2 dimensions.

        The first index points to the particular sample and the second index

        points to an intensity at some time for some regularized decay.

        """
        #decay_input = ['ND', 'NB']
        ND_tensor = self.training_tensor_proc[idx]
        target = self.target_tensor_proc[idx]

        # print(multiple_decays)
        return ND_tensor, target
        
        
class ND_Reg(Dataset):
    def __init__(self, lam, true_Signal, set_type, ND_tensor, recon_tensor, g_truths, estimates_tensor, oracle_curves, lam_oracle, NN_params, oracle_parameters, mean1 = None, std1 = None, convolutional = True):
        print('BEGIN ND_REG!!!')
        self.estimates = estimates_tensor
        self.true_signal = true_Signal
        self.ND = ND_tensor
        self.ld = lam
        self.oracle_lam = lam_oracle
        self.oracle_curve = oracle_curves
#         self.pert_lam = lam_pert
#         self.pert_signals = pert_signals
#         self.pert_params = pert_params
        self.estim_params = NN_params
        self.oracle = oracle_parameters
        if convolutional:
            self.training_tensor_proc = torch.stack([ND_tensor, recon_tensor], dim = 1)
            
            if set_type == "Training":
                self.mean2, self.stdev = torch.tensor([0.0]), torch.tensor([1.0])
#                 self.training_tensor_proc.mean([0,2]), self.training_tensor_proc.std([0,2])
                self.training_tensor_proc = ((self.training_tensor_proc - self.mean2[None,:,None])/(self.stdev[None,:,None])).float()
                self.target_tensor_proc = g_truths.float() 
#                 torch.from_numpy(mm.fit_transform(t2_truths))
            else:
                self.training_tensor_proc = ((self.training_tensor_proc - mean1[None,:,None])/(std1[None,:,None])).float()
                self.target_tensor_proc = g_truths.float() 
#                 torch.from_numpy(mm.transform(t2_truths))
                
#             t21_tensor = torch.tensor(self.master_frame["T21_t"])
#             t22_tensor = torch.tensor(self.master_frame["T22_t"])
        
    
    
    def __len__(self):
        #decay_input = ['ND', 'NB']
        """

        Returns the number of samples in the data set

        """

        return self.training_tensor_proc.shape[0]

    def __getitem__(self, idx):

        """

        Returns samples at idx.

        - If idx is an int, the output is a pair of 1-d Tensors: the first is

        the regularized decays stacked horizontally and the second is the

        corresponding target parameters.

        - If idx is a list, it returns two Tensors, each with 2 dimensions.

        The first index points to the particular sample and the second index

        points to an intensity at some time for some regularized decay.

        """
        #decay_input = ['ND', 'NB']
        ND_Reg_tensor = self.training_tensor_proc[idx]
        target = self.target_tensor_proc[idx]

        # print(multiple_decays)
        return ND_Reg_tensor, target


def initDataset(pkl_file, set_type = None, select_target = "Lambda", type1 = "full", trueSig_Data = None, mean = None, std = None, mean_targ = None, std_targ = None, convolutional = False):
    if type1 == "full":
        return Lambda_FULL(pkl_file)
    elif type1 == "standardized":
        return Lambda_Standardized(pkl_file, set_type, select_target, mean, std, mean_targ, std_targ, convolutional)
    elif type1 == "trueSigGen":
        return Lambda_trueSigGen(pkl_file, set_type, select_target, trueSig_Data, mean, std, convolutional)
    else:
        return Lambda(pkl_file)
    

def toDataset(dp, estim_path, PE_path, means = None, stdevs = None, CNN = True):
   
    dataPurpose = dp
#     if dataPurpose == "Testing":
#         T22_high = 475.0

#     elif dataPurpose == "Validation":
#         T22_high = 475.0

#     elif dataPurpose == "Training":
#         T22_high = 500.0

#     time_low, time_high = 0.0, 1.6 * T22_high
    times = np.linspace(8, 256, 32)
#     np.linspace(time_low, time_high, 64, dtype=np.float64)

    print(f"Building {dp} Dataset...")


    df = pd.read_feather(estim_path)
    df["Index"] = df["Index"].astype(int)
    df.set_index("Index", inplace = True, drop = True)
    df.sort_index(inplace = True)
    
#     df_pert = pd.read_feather(perturbed_path)
#     df_pert["Index"] = df_pert["Index"].astype(int)
#     df_pert.set_index("Index", inplace = True, drop = True)
#     df_pert.sort_index(inplace = True)

    convolutional = CNN

    df3 = pd.read_feather(PE_path)

    T2_values = df[["t21_ld", "t22_ld"]].values

    c1_ld = df["c1_ld"].values
    lam = df["lambda"].values
#     lam_pert = df_pert["lambda"].values
#     pert_params = np.array(df_pert[["t21_ld", "t22_ld", "c1_ld"]].values)
#     pert_signals = myTrueModel(times.reshape(1,-1), pert_params[:,2].reshape(-1,1), pert_params[:,0].reshape(-1,1), pert_params[:,1].reshape(-1,1))

    NN_params = np.array(df[["t21_ld", "t22_ld", "c1_ld"]])

    

    lam_oracle = df3["Lambda"].values

    

#     assert(df3["Lambd
#     combined = T2_values

#     mask = np.greater(combined[:, 0], combined[:, 1])

#     combined[mask] = np.stack([combined[mask,1], combined[mask,0]], axis = 1)

#     c1_ld[mask] = 1.0 - c1_ld[mask]

    recon_signals = myTrueModel(times.reshape(1,-1), c1_ld.reshape(-1,1), T2_values[:,0].reshape(-1,1), T2_values[:,1].reshape(-1,1))
    print("Recon Signal: ", recon_signals.shape)

    estimates_tensor = torch.from_numpy(T2_values)
    recon_tensor = torch.from_numpy(recon_signals)
    ND_signals = torch.from_numpy(np.stack(df3["Noisy Signal"].values))
    print(ND_signals.shape)
    regTraj_noisySignals = torch.from_numpy(np.stack(df["ND"].values))
    print(regTraj_noisySignals.shape)
    bool = torch.equal(regTraj_noisySignals.float(), ND_signals.float())
    print("IDX BASED? ", bool)
    assert(bool)
    
    g_truths = torch.from_numpy(df3[["T21_t", "T22_t", "c1_t"]].values)

    oracle_parameters = torch.from_numpy(df3[["T21_est", "T22_est", "c1_est"]].values)
    oracle_curves = myTrueModel(times.reshape(1,-1), oracle_parameters[:,2].reshape(-1,1), oracle_parameters[:,0].reshape(-1,1), oracle_parameters[:,1].reshape(-1,1))

    true_Sig = myTrueModel(times.reshape(1,-1), g_truths[:,2].reshape(-1,1), g_truths[:,0].reshape(-1,1), g_truths[:,1].reshape(-1,1))
    
#     ts_tensor = torch.from_numpy(true_Sig)
    
    if dp == "Training":
        dS = ND_Reg(lam, true_Sig, dp, ND_signals, recon_tensor, g_truths, estimates_tensor, oracle_curves, lam_oracle, NN_params, oracle_parameters, convolutional = convolutional)
    else:
        
        dS = ND_Reg(lam, true_Sig, dp, ND_signals, recon_tensor, g_truths, estimates_tensor, oracle_curves, lam_oracle, NN_params, oracle_parameters, mean1 = means, std1 = stdevs, convolutional = convolutional)
    print("DATASET: ", (dS.training_tensor_proc).shape)
    return dS


def toDataset_REG(dp, reg_path, PE_path, means = None, stdevs = None, CNN = True):
   
    convolutional = CNN
    print(f"Building {dp} Dataset...")

    df3 = pd.read_feather(PE_path)

    recon_signals = torch.load(reg_path)
    # myTrueModel(times.reshape(1,-1), c1_ld.reshape(-1,1), T2_values[:,0].reshape(-1,1), T2_values[:,1].reshape(-1,1))
    print("Recon Signal: ", recon_signals.shape)

    # estimates_tensor = torch.from_numpy(T2_values)
    recon_tensor = recon_signals
    ND_signals = torch.from_numpy(np.stack(df3["Noisy Signal"].values))
    # bool = torch.equal(torch.from_numpy(np.stack(df["ND"].values)), ND_signals)
    print("IDX BASED? ", bool)
    assert(bool)
    
    g_truths = torch.from_numpy(df3[["T21_t", "T22_t", "c1_t", "Lambda"]].values)

    # true_Sig = myTrueModel(times.reshape(1,-1), g_truths[:,2].reshape(-1,1), g_truths[:,0].reshape(-1,1), g_truths[:,1].reshape(-1,1))
    
#     ts_tensor = torch.from_numpy(true_Sig)
    
    if dp == "Training":
        dS = ND_Reg(None, None, dp, ND_signals, recon_tensor, g_truths, None, convolutional = convolutional)
    else:
        
        dS = ND_Reg(None, None, dp, ND_signals, recon_tensor, g_truths, None, mean1 = means, std1 = stdevs, convolutional = convolutional)
    print("DATASET: ", (dS.training_tensor_proc).shape)
    return dS


