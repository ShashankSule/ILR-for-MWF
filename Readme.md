# ILR for MWF

## Introduction
This repository contains the implementation of the hyperparameter-tuned input layer regularization (ILR) networks for myelin water estimation. The ILR model is designed to produce additionally accurate estimates of the MWF by augmenting noisy signals with smoothened or regularized versions. 

Below is an overview of the ILR + hyperparameter tuning architecture:

![ILR Architecture](graphical_abstract.pdf)

## Initial steps

First, you are required to move data into this repository from [here](3pe.yml). Note that this link contains *all* of the data generated for our experiments, so if you wish to simply analyse the data and networks that we have trained, you can skip the next section on training and generating data. If you wish to regenerate the data, please also set up a new python environment with the provided `yml` file: 

``` conda env create -f 3pe.yml ```

## Training and Generating Data
The training pipeline and data generation process are handled by the scripts in the `src/` folder. We have three elements to our pipeline: (1) generating approximations of the oracle lambdas via either NNs or GCV, (2) augmenting noisy biexponential data with a smooth part arising from such lambda, and (3) training the (ND, Reg) networks. Initially, the script `src/generate_data.py` may be used to generate noisy biexponential signals. From here, we partition into two types of workflows: 

1. Training $(ND, Reg)_{NN}$:

The networks $\lambda_{NN}$ can be trained in `src/train_lambda_NN.py`. Next, the concatenated data is generated in `src/generate_NDReg_NN_data.py`. Finally, the ILR networks are trained in `src/train_ND_Reg_NN.py`. 

2. Training $(ND, Reg)_{GCV}$: 

Since there is no training involved in generating $\lambda_{GCV}$, steps (1), (2), and (3) can be incorporated in one script `src/GCV_Rician_Training_GenerateRegAndTrain.py`. 


## Visualizing Results
Results can be visualized using the Jupyter notebooks included in the repository. The figures generated from the trained ILR model can be found in the following notebooks:

- [Notebook for analyzing lambda selection](/tutorials/Lambda_comparisons.ipynb)
- [Notebook for evaluating ILR performance](/tutorials/Param_estimation_synthetic.ipynb)
- [Analysis of brain data](/tutorials/BrainData.ipynb)

