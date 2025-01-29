import numpy as np

import torch

import torch.nn as nn





##-----------------------------------------------------------------##

## Functions that give a pure signal, and another that adds noise  ##

##-----------------------------------------------------------------##



## A) True signal - Three Parameters

#----------------

def myTrueModel(t, c, t21, t22, *, signalType="biexponential"):

    """

    Produces a pure, uncorrupted signal whose parameters we aim to approximate.



    Input:

    ------

        t (Array of length num_times) - Signal acquisition times.  

        c (Nonnegative float) - initial fraction of component one.  

        t21 (Nonnegative float) - Time constant for component 1.

        t22 (Nonnegative float) - Time constant for component 2.

        signalType (string, optional kwarg) - Type of signal. Choices are 

            biexponential decay, power, quadratic, or sinusoidal.



    Output:

    -------

        signal (Array of length num_times) - Pure signal of the specified type

"""

    if signalType=="biexponential":

        return c*np.exp(-t/t21) + (1.0-c)*np.exp(-t/t22)

    elif signalType=="power":

        return c + (t/t21)**t22

    elif signalType=="quadratic":

        return c + (t/t21) + (t/t22)**2

    elif signalType=="sinusoidal":

        return c*np.cos(t/t21) + (1.0-c)*np.cos(t/t22)



## A) True signal - Two Parameters

#---------------------------------



def myTrueModel_2param(t, c1, c2, t21, t22, *, signalType="biexponential"):

    """

    Produces a pure, uncorrupted signal whose parameters we aim to approximate.



    Input:

    ------

        t (Array of length num_times) - Signal acquisition times.

        c1 (Nonnegative float) - initial fraction of component one.

        c2 (Nonnegative float) - initial fraction of component two.

        t21 (Nonnegative float) - Time constant for component 1.

        t22 (Nonnegative float) - Time constant for component 2.

        signalType (string, optional kwarg) - Type of signal. Choices are

            biexponential decay, power, quadratic, or sinusoidal.



    Output:

    -------

        signal (Array of length num_times) - Pure signal of the specified type

"""

    if signalType=="biexponential":

        return c1*np.exp(-t/t21) + (c2)*np.exp(-t/t22)

    elif signalType=="power":

        return c1 + (t/t21)**t22

    elif signalType=="quadratic":

        return c1 + (t/t21) + (t/t22)**2

    elif signalType=="sinusoidal":

        return c1*np.cos(t/t21) + (c2)*np.cos(t/t22)



## B) Noisy signal

#-----------------

def myNoisyModel(signal, snr, *, signalType="biexponential"):

    """

    Adds Gaussian noise to a pure signal. In particular, components of the 

    noise are independent, identically distributed normal random variables

    with mean 0.0 and standard deviation that depends on the signalType.

    For biexponential and sinusoidal signals, the standard deviation == 

    1.0/sqrt(snr).  For power and quadratic signals, the standard deviation

    ==  sqrt[(time average of the signal)/snr].



    Input:

    ------

        signal (Tensor of size (batch_size, num_times)) - Tensor of pure 

            signal.  Rows index sample and columns index time.

        snr (positive float) - Signal-to-noise ratio

        signalType (string, optional kwarg) - Type of signal. Choices are 

            biexponential decay, power, quadratic, or sinusoidal.



    Output:

    -------

        noisy_signal (Tensor of size (batch_size, num_times)) - Tensor of

            noisy signals.  Rows index sample and columns index time.

    """

    if type(signal) == np.ndarray:

        num_times  = len(signal)

        num_signals = 1



        if (signalType == "biexponential") or (signalType == "sinusoidal"):
            noiseLevel=np.max(signal)/snr
            # noiseLevel = 1.0/snr

            noise_Amp_on_SC = np.random.normal(loc=0.0, scale=noiseLevel, size=num_times)#eps_prime
            noise_on_QC = np.random.normal(loc=0.0, scale=noiseLevel, size=num_times) #eps_~

    elif type(signal) == torch.Tensor:
        num_times   = signal.shape[0]

        try:
            num_signals = signal.shape[1]
        except:
            num_signals = 1

        if (signalType == "biexponential") or (signalType == "sinusoidal"):

            noiseLevel=torch.max(signal)/snr

            noise_Amp_on_SC = torch.normal(mean=0.0, std=noiseLevel, size = num_times) 
            noise_on_QC = torch.normal(mean=0.0, std=noiseLevel, size = num_times)

    return ((signal + noise_Amp_on_SC)**2 + noise_on_QC**2)**0.5







    



class Signal(object):

    def __init__(self,time_low, time_high, num_times, SNR, signalType, dataType="Tensor"):



        # What type of signal == being modeled? "biexponential", "sinusoidal", etc.

        self.signalType = signalType



        # Signal-to-noise ratio

        self.snr = SNR



        # Times at which the signal's intensity == measured. "acquisition times"

        if dataType == "Tensor":

            self.times = torch.linspace(time_low, time_high, num_times, dtype=torch.float64)

        else:

            self.times = np.linspace(time_low, time_high, num_times)



        # Define the equation for the true signal model.

        if self.signalType == "biexponential":

            def biexponential_decay_np(c, T21, T22):

                return c*np.exp(-self.times/T21) + (1.0-c)*np.exp(-self.times/T22)                

            def biexponential_decay_tensor(c, T21, T22):

                return c*torch.exp(-self.times/T21) + (1.0-c)*torch.exp(-self.times/T22)                

            if dataType == "Tensor": self.true_model = biexponential_decay_tensor

            else: self.true_model = biexponential_decay_np



        elif self.signalType == "power":

            def power_law(c, T21, T22):

                return c + (self.times/T21)**T22

            self.true_model = power_law



        elif self.signalType == "quadratic":

            def quadratic_signal(c, T21, T22):

                return c + (self.times/T21) + (self.times/T22)**2

            self.true_model = quadratic_signal



        elif self.signalType == "sinusoidal":

            def sinusoidal_signal_np(c, T21, T22):

                return c*np.cos(self.times/T21) + (1.0-c)*np.cos(self.times/T22)

            def sinusoidal_signal_tensor(c, T21, T22):

                return c*torch.cos(self.times/T21) + (1.0-c)*torch.cos(self.times/T22)

            if dataType == "Tensor": self.true_model = sinusoidal_signal_tensor

            else: self.true_model = sinusoidal_signal_np



        # Define how noise should be added to the signal.

        if dataType == "numpy":



            if (self.signalType == "biexponential") or (self.signalType == "sinusoidal"):

                def noise_model(signal):

                    assert (len(self.times)==len(signal)), (f"Number of times" +

                    "({len(self.times)}) and length of signal ({len(signal)})" + 

                    "are not equal.")

            

                    num_times  = len(self.times)

                    num_signals = 1



                    noiseLevel = 1.0/self.snr

                    noise = np.random.normal(loc=0.0, scale=np.sqrt(noiseLevel), size=num_times)

                    return signal + noise



            elif (self.signalType == "power") or (self.signalType == "quadratic"):

                def noise_model(signal):

                    assert (len(self.times)==len(signal)), (f"Number of times" +

                    "({len(self.times)}) and length of signal ({len(signal)})" + 

                    "are not equal.")

            

                    num_times  = len(self.times)

                    num_signals = 1



                    avg_signal = np.mean(signal)

                    noiseLevel = avg_signal/self.snr

                    noise = np.random.normal(loc=0.0, scale=np.sqrt(noiseLevel), size=num_times)

                    return signal + noise



        elif dataType == "Tensor":



            if (self.signalType == "biexponential") or (self.signalType == "sinusoidal"):

                def noise_model(signal):

                    num_times   = len(signal[0,:]) # assumes all signals have same # of measurements

                    assert (len(self.times)==num_times), (f"Number of times" +

                    "({len(self.times)}) and length of signal ({num_times})" + 

                    "are not equal.")



                    num_signals = len(signal[:,0])



                    noiseLevel = 1.0/self.snr

                    noise = torch.cat( [ torch.normal(mean=0.0,

                                                      std=np.sqrt(noiseLevel),

                                                      size=(1,num_times))

                                         for _ in range(0,num_signals) ]

                                       ,0)

                    return signal + noise

                

            elif (self.signalType == "power") or (self.signalType == "quadratic"):

                def noise_model(signal):

                    num_times   = len(signal[0,:]) # assumes all signals have same # of measurements

                    assert (len(self.times)==num_times), (f"Number of times" +

                    "({len(self.times)}) and length of signal ({num_times})" + 

                    "are not equal.")



                    num_signals = len(signal[:,0])



                    avg_signal = torch.mean(signal, dim=1, keepdim=True)

                    noiseLevel = avg_signal/self.snr

                    noise = torch.cat( [ torch.normal(mean=0.0,

                                                      std=torch.sqrt(noiseLevel[i,0]),

                                                      size=(1,num_times))

                                         for i in range(0,num_signals) ]

                                       ,0)

                    return signal + noise

        self.add_noise = noise_model

        print(f'The noise model == {self.add_noise}')









        

