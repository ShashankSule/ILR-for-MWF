import csv
from hashlib import new

from itertools import product, zip_longest
import multiprocessing

import numpy as np

import os

import pandas as pd

import sys

import torch

from tqdm import tqdm

# from Reordering_Swapping_for_GPU import parameter_swap_same_tensor

from makeSignals import myTrueModel, myNoisyModel, myTrueModel_2param

from regTraj import least_squares_2param, least_squares_3param, curve_fit_2param, make_l2_trajectory

# from writeParams import writeSummary

import multiprocess as mp

from multiprocessing import Pool, freeze_support
from multiprocessing import set_start_method
import torch.multiprocessing as mpp

# import time as time2
# import h5py

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import config
from GSS import fmin_bound_norm


def sampleSolver(tens):
#     print(tens)
    c1_t, T21_t, T22_t, j = tens    
    c2_t = 1.0 - c1_t

    # Swapping target values. Enforces T21 <= T22.
#     prn
    if config.num_params == 3:
        
        true_signal = myTrueModel(config.time, c1_t, T21_t, T22_t, signalType=config.mySignalType)
        
    
    lambdaset = []
    for k in range(config.noise_realizations):

        # sample_idx = j * config.noise_realizations + k
        # sample_idx
                
        try: 
            noisy_signal = myNoisyModel(true_signal, config.SNR, signalType=config.mySignalType).squeeze(0)
        except:
            noisy_signal = myNoisyModel(true_signal, config.SNR, signalType=config.mySignalType)
        
        noisy_signal = noisy_signal/noisy_signal[0]

        new_sample_data_frame = pd.DataFrame(columns=["Noisy Signal", "Lambda", "c1_t", "c1_est", "T21_est", "T22_est", "T21_t", "T22_t"])
        
        new_sample_data_frame["Noisy Signal"] = [noisy_signal] # noisy signals, 1 lbda per noisy signal. 

        
        if config.num_params == 3:
#             print("Starting")
            c1_ld, T21_ld, T22_ld, min_error_lambdas =  fmin_bound_norm(noisy_signal, c1_t, c2_t, T21_t, T22_t)



#             recon_signal = myTrueModel(config.time, c1_ld, T21_ld, T22_ld, signalType=config.mySignalType)

        

        #Swapping reg traj:
            

        new_sample_data_frame["Lambda"] = 10**min_error_lambdas
        new_sample_data_frame["c1_t"] = c1_t
        # new_sample_data_frame["c2_t"] = c2_t
        new_sample_data_frame["c1_est"] = c1_ld
        new_sample_data_frame["T21_est"] = T21_ld
        new_sample_data_frame["T22_est"] = T22_ld
        new_sample_data_frame["T21_t"] = T21_t
        new_sample_data_frame["T22_t"] = T22_t
        # new_sample_data_frame["TrueSignal"] = [true_signal]
        # new_sample_data_frame["Recon Signal"] = [recon_signal]
        
        lambdaset.append(new_sample_data_frame)
#         print("Finishing...")
#     print("done")
    return pd.concat(lambdaset, ignore_index=True)