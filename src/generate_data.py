import csv

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

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import config
import noisySignalGen_3



if __name__ == '__main__':
    freeze_support()
    mpp.set_start_method('spawn', force=  True)


    num_cpus_avail = 64
    print("Using Super Computer")

    # else:
    #     num_cpus_avail = mpp.cpu_count()//2
    SNRs = [100.0]
    for snr_v in SNRs:
        DATASETS = ["Testing","Validation"]
        for dataset_type in DATASETS:
            config.initializer(dataset_type, snr_v)
            print(f"Building {dataset_type} Dataset")

            lis = []
    #         counter=0
            with mp.Pool(processes = num_cpus_avail, initializer = config.initializer, initargs = (dataset_type,snr_v,)) as pool:
                with tqdm(total=config.tensor_targ_iter.shape[0]) as pbar:
                    for item in pool.imap_unordered(noisySignalGen_3.sampleSolver, config.tensor_targ_iter, chunksize=1):
    #                     print("Returning...")
    #                     count+=1
    #                     print(count)
                        lis.append(item)
                        pbar.update()

                pool.close() #figure out how to implement
                pool.join()

            training_data_frame = pd.concat(lis, ignore_index= True)
            # training_data_frame["TE"] = [config.time]*len(training_data_frame.index)

            print(training_data_frame.shape) #should be num_triplets X num_realizations
            training_data_frame.to_feather("Lambda_TrainingData/LambdaTraining/" + config.thisDatasName)