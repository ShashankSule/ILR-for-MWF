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

# Data visualization of NN
import pandas as pd
from tqdm import tqdm

#Testing Files
from makeSignals import myTrueModel
from scipy.optimize import curve_fit
from scipy.optimize import fminbound
from scipy import special
from multiprocessing import freeze_support
import multiprocess as mp
from torch.utils.data import Dataset


GENERATE = False
TRAIN = True





if GENERATE:
    low_power = -7    #Low power of lambda used
    high_power = 3   #High power of lambda used



    def mycurvefit_l2Regularized_3param(d, initial_est, ld, signalType="biexponential", lb_T21=0.0, lb_T22=0.0, lb_c1=0.0, ub_T21=np.inf, ub_T22=np.inf, ub_c1=np.inf): #c1 ub = 1.0??


        SNR = snr
        times = np.linspace(8.0, 256.0, 32)

        D = np.array([1.0,  # T2,1

                    1.0,  # T2,2

                    100.0])  # C1


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


        t_dim = times.ndim

        indep_var = np.concatenate((times,

                                    np.array(ld,ndmin=t_dim)))



        d_dim = d.ndim

        depen_var = np.concatenate((d, np.array(0.0, ndmin=d_dim), np.array(0.0,ndmin=d_dim), np.array(0.0, ndmin=d_dim)))


        try:
            opt_val = curve_fit(expectation_Rice, indep_var, depen_var,  # curve, xdata, ydata

                                p0=initial_est,  # initial guess

                                bounds=([lb_T21, lb_T22, 0.0], [ub_T21, ub_T22, 1.0]),

                                method="trf",

                                max_nfev=5000)
            #print('!!!!!!!!!!', opt_val)

        except RuntimeError:
            opt_val = (np.asarray([(1.0+50.0)/2,(40.0+225.0)/2,(0.0+0.60)/2]), np.asarray([(1.0+50.0)/2,(40.0+225.0)/2,(0.0+0.60)/2]))

            # print("maximum number of function evaluations == exceeded")





            # returns estimate. second index in estimated covariance matrix

        T21_ld, T22_ld, c1_ret = opt_val[0]
        T21_ld = T21_ld.item()
        T22_ld = T22_ld.item()
        c1_ret = c1_ret.item()

        if T21_ld > T22_ld:
            T21_ld_new = T22_ld
            T22_ld = T21_ld
            T21_ld = T21_ld_new
            c1_ret = 1.0 - c1_ret
            assert (T21_ld != T22_ld)

        return c1_ret, T21_ld, T22_ld

    def G(t, con_1, tau_1, tau_2):
        function = con_1*np.exp(-t/tau_1) + (1.0-con_1)*np.exp(-t/tau_2)
        return function


    def J(t, con1, tau1, tau2):
        func1 = np.exp(-t/tau1) - np.exp(-t/tau2)
        func3 = (con1*t)*np.exp(-t/tau1)/(tau1**2)
        func4 = ((1.0-con1)*t)*np.exp(-t/tau2)/(tau2**2)

        jacobian = np.stack((func1, func3, func4), axis=-1)
        return jacobian

    def get_GCV_value(noisey_data, initial_guess, lamb):

        c1_ld, t21_ld, t22_ld = mycurvefit_l2Regularized_3param(noisey_data, initial_guess, lamb)
        reg = G(np.linspace(8.0, 256.0, 32), c1_ld, t21_ld, t22_ld)
        RSS = np.sum((reg - noisey_data)**2)


        wmat = np.array([[100.0,0,0],[0,1.0,0],[0,0,1.0]])
        GCVjacobian = J(np.linspace(8.0, 256.0, 32), c1_ld, t21_ld, t22_ld)
        C_GCV = GCVjacobian@np.linalg.inv(GCVjacobian.transpose()@GCVjacobian+(lamb**2)*wmat.transpose()@wmat)@GCVjacobian.transpose()
        (n,n) = C_GCV.shape
        identity = np.identity(n)
        GCVdenominator = (identity - C_GCV).trace()
        GCV = RSS/(GCVdenominator**2)
        return GCV


    def fmin_bound_norm(i):
        noisey_data = data[i][:32]
        p_0_e = data[i][32:]
    #     print("starting")
        lamb_pow = fminbound(lambda lamb: get_GCV_value(noisey_data, p_0_e, 10**lamb), low_power, high_power, xtol = 10**-3)
    #     print("done")
        return *mycurvefit_l2Regularized_3param(noisey_data, p_0_e, 10**lamb_pow), lamb_pow, i


    for dp in ["Training", "Validation", "Testing"]:
        for snr in [5.0, 50.0, 100.0]:
            curr_path = os.path.abspath('')
            testing_path =  os.path.relpath(f"../Lambda_TrainingData/LambdaGeneration/DenseGraph_ExpectationRicianNoise_1000NR_SNR_{snr}_{dp}Data.feather",curr_path)

            # convolutional = True
            testing_dataset = pd.read_feather(testing_path)
            data_NDs = np.stack(testing_dataset["Noisy Signal"].values)
            ground_truths_all = np.array(testing_dataset[["Lambda","c1_t", "T21_t", "T22_t"]].values, dtype = np.float64)
            p_0_s = np.array(testing_dataset[["T21_t","T22_t", "c1_t"]].values, dtype = np.float64)

        #     raw_data_path = "rS_slice5.mat"
        # #     "NESMA_slice5.mat"
        #     raw_data = scipy.io.loadmat(raw_data_path)
        #     raw_signals = raw_data["slice_oi"]
        #     mask = 700
        #     masking_idxs = raw_signals[:,:,0] > mask
        #     # pixels_retained = np.argwhere(masking_idxs)
        #     ND_signals = raw_signals[masking_idxs]
        #     raw_norm = ND_signals/ND_signals[:,0][:,None]

            data_all = np.concatenate((data_NDs,p_0_s), axis = 1)

            unique_c1s = np.unique(ground_truths_all[:,1])

            print(unique_c1s)

            for unique_val in unique_c1s:
                idxs = np.where(ground_truths_all[:,1] == unique_val)[0]
                data = data_all[idxs]
                ground_truths = ground_truths_all[idxs]
                # display(idxs)

                print(data.shape)


                print("Starting MP...")

                # data = raw_norm


                if __name__ == '__main__':
                    freeze_support()
                    print("Finished Assignments...")

                    num_cpus_avail = 80
                    print("Using Super Computer")

                    print(f"Building GCV Dataset...")
                    lis = []

                    with mp.Pool(processes = num_cpus_avail) as pool:

                        with tqdm(total=data.shape[0]) as pbar:
                            for c1, t21, t22, ld, k in pool.imap_unordered(fmin_bound_norm, range(data.shape[0]), chunksize = 50):
                                # if k == 0:
                                    # print("Starting...")
                                lis.append(np.concatenate([np.array([idxs[k]]), ground_truths[k],np.array([c1, t21, t22, k, ld]), data[k][:32]], dtype=np.float64))

                                pbar.update()
                    #             # break
                        pool.close() #figure out how to implement
                        pool.join()
                    # # assert False
                    T2_ld = np.stack([row for row in lis], axis = 0)
                    print(T2_ld.shape) #should be num_triplets X num_realizations
                    # assert(T2_ld.shape == )
                    df = pd.DataFrame(index = range(T2_ld.shape[0]), columns = ["high_order_idxs", "Lambda","c1_t", "T21_t", "T22_t","c1_ld","t21_ld", "t22_ld", "Index", "lambda_pred", "ND"])
                    df[["high_order_idxs","Lambda","c1_t", "T21_t", "T22_t", "c1_ld", "t21_ld", "t22_ld", "Index", "lambda_pred"]] = T2_ld[:,:-32]
                    df["ND"] = [T2_ld[i,-32:] for i in range(T2_ld.shape[0])]

                    print("DATAFRAME: ", df.shape)
                    # df[df.columns] = T2_ld
                    # df.set_index("Index", inplace=True, drop = True)
                    # df.sort_index(inplace = True)
            #             if option == 0:
            #                 df.to_feather(f"../Lambda_TrainingData/3PE_ReconSignals/3PE_ReconSignals_Lambda0_SNR_900.0_{dp}Data.feather")
            #             elif option ==1:
                    df.to_feather(f"../Lambda_TrainingData/3PE_ReconSignals/LambdaGCV_{dp}/NoOffset_Sectionc1_{np.round(unique_val,4)}_3Parameter_LambdaGCVs_{dp}_SNR_{snr}_SyntheticRicianNoise.feather")
                    # WR = (f"../Lambda_TrainingData/3PE_ReconSignals/3PE_ReconSignals_WR_128Pts_SNR_900.0_{dp}Data.feather")
                    # myelin = f"../Lambda_TrainingData/3PE_ReconSignals/3PE_ReconSignals_LambdaNN_Myelin_128Pts_SNR_900.0_{dp}Data.feather"
                    # "../Lambda_TrainingData/3PE_ReconSignals/3PE_BrainDataSignals_Myelin_32Pts_SNR_900.0_{dp}Data.feather"
                    
class ND_Reg(Dataset):
    def __init__(self, lam, true_Signal, set_type, ND_tensor, recon_tensor, g_truths, estimates_tensor, oracle_curves, lam_oracle, NN_params, oracle_parameters, mean1 = None, std1 = None, convolutional = True):
        print('BEGIN ND_REG!!!')
        self.estimates = estimates_tensor
        self.true_signal = true_Signal
        self.ND = ND_tensor
        self.ld = lam
        self.oracle_lam = lam_oracle
        # self.oracle_curve = oracle_curves
#         self.pert_lam = lam_pert
#         self.pert_signals = pert_signals
#         self.pert_params = pert_params
        self.estim_params = NN_params
        # self.oracle = oracle_parameters
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

if TRAIN:
    if __name__ == "__main__":
        
        curr_path = os.path.abspath('')
        
        def gen_names(snr, testing_path, dp):

            testing_dataset = pd.read_feather(testing_path)
            ground_truths_all = np.array(testing_dataset[["Lambda","c1_t", "T21_t", "T22_t"]], dtype = np.float64)

            unique_c1s = np.unique(ground_truths_all[:,1])
            return [f"../Lambda_TrainingData/3PE_ReconSignals/LambdaGCV_{dp}/NoOffset_Sectionc1_{np.round(val,4)}_3Parameter_LambdaGCVs_{dp}_SNR_{snr}_SyntheticRicianNoise.feather" for val in unique_c1s], testing_dataset

        
        for snr in [5.0,50.0,100.0]:
            print(f"=========================SNR: {snr}===================================")
            for mod_type in ["NDReg_GCV"]:
                dS_Dict = {"Training" : None, "Validation": None}
                # mm = MinMaxScaler()
                means = None
                stdevs = None

                for dp in ["Training", "Validation"]:
                        
                    testing_path =  os.path.relpath(f"../Lambda_TrainingData/LambdaGeneration/DenseGraph_ExpectationRicianNoise_1000NR_SNR_{snr}_{dp}Data.feather",curr_path)
                    
                    file_list, base_df = gen_names(snr, testing_path, dp)
                    df_list = [pd.read_feather(file) for file in file_list]
                    master_df = pd.concat(df_list, axis=0, ignore_index=True)
                    print(base_df.head(5))

                    master_df["high_order_idxs"] = master_df["high_order_idxs"].astype(int)
                    master_df.set_index("high_order_idxs", inplace = True, drop = True)
                    master_df.sort_index(inplace = True)
                    print(master_df.head(5))

                    base_NDs = np.stack(base_df["Noisy Signal"].values)
                    master_NDs = np.stack(master_df["ND"].values)
                    bool = (np.round(np.float64(base_NDs), 4) == np.round(np.float64(master_NDs), 4)).all()
                    assert(bool)
                    print(f"IDX BASED? {bool}")
                    
                    
                    times = np.linspace(8.0, 256.0, 32)

                    T2_values = master_df[["t21_ld", "t22_ld"]].values
                    c1_ld = df["c1_ld"].values
                    lam = df["lambda_pred"].values
                    NN_params = np.array(df[["t21_ld", "t22_ld", "c1_ld"]])
                    lam_oracle = base_df["Lambda"].values

                    recon_signals = myTrueModel(times.reshape(1,-1), c1_ld.reshape(-1,1), T2_values[:,0].reshape(-1,1), T2_values[:,1].reshape(-1,1))
                    print("Recon Signal: ", recon_signals.shape)

                    estimates_tensor = torch.from_numpy(T2_values)
                    recon_tensor = torch.from_numpy(recon_signals)
                    ND_signals = torch.from_numpy(master_NDs)
                    
                    g_truths = torch.from_numpy(master_df[["T21_t", "T22_t", "c1_t"]].values)

                    # oracle_parameters = torch.from_numpy(df3[["T21_est", "T22_est", "c1_est"]].values)
                    # oracle_curves = myTrueModel(times.reshape(1,-1), oracle_parameters[:,2].reshape(-1,1), oracle_parameters[:,0].reshape(-1,1), oracle_parameters[:,1].reshape(-1,1))

                    true_Sig = myTrueModel(times.reshape(1,-1), g_truths[:,2].reshape(-1,1), g_truths[:,0].reshape(-1,1), g_truths[:,1].reshape(-1,1))
                    
                    means, stdevs = None, None
                    if dp !="Training":
                        means = torch.tensor([0])
                        stdevs = torch.tensor([1])

                    dS_Dict[dp] = ND_Reg(lam, true_Sig, dp, ND_signals, recon_tensor, g_truths, estimates_tensor, None, lam_oracle, NN_params, None, convolutional = True)
                

                
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

                       
                    DATA_NAME = f"3P_{mod_type}__GCV__NDReg__SNR_{snr}"
                    
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