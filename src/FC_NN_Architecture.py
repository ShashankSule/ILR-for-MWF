import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

# def scaled_customLoss(predictions, targets, weighting):

class LambdaLoss():
    def __init__(self, device, eps = 1e-2, scaling = 100.0, times = None):
        # pass
#         super().__init__()
#         self.dev = device
        self.scale = torch.tensor([1.0,1.0,scaling], dtype = float, device = device)
#         self.weight_hP = torch.tensor([1,1, 1/150], dtype = float).to(device)
        self.epsilon = torch.tensor([0.0,0.0,eps], dtype = float, device = device)
        if times != None: 
            self.time_new = times.view(1,-1)


    def LogLoss(self, pred, actual):
#         diff_matrix = torch.zeros(actual.shape, dtype=float).to(self.dev)
#         idxs = actual[:,2] < 0.20 #Where c1_truth < 0.20 penalize highly (self.weight_hP)
#         diff_matrix[idxs] = ((pred[idxs] - actual[idxs])/self.weight_hP)**2
#         diff_matrix[~idxs]= ((pred[~idxs] - actual[~idxs])/self.weight_lP)**2
        loss = ((torch.log10(pred + 1e-9) - torch.log10(actual + 1e-9))**2).mean()
        return loss
    
    def penaltyNegVal(self, pred, actual):
        loss = (((pred-actual)*self.scale)**2).mean() + (1e4*((torch.abs(pred) - pred).mean(0))).sum()
        return loss

    def MSELoss(self, pred, actual):
        loss = (((pred-actual)*self.scale)**2).mean()
        return loss
    
    def L1Loss(self, pred, actual):
        loss = (torch.abs(pred-actual)*self.scale).mean()
        return loss
    
    def c1Loss(self, pred, actual):
        loss = (((pred-actual)*self.scale[-1])**2).mean()
        return loss
    
    def signal_loss(self, pred_param, actual_param, pred_denoise, true_signal):
        loss = (((pred_param-actual_param)*self.scale)**2).mean() + 1e4*((((pred_denoise-true_signal))**2).mean())
        return loss
    
    def biexp_loss(self, pred, actual):
        with torch.set_grad_enabled(True):
            t21_pred, t22_pred, c1_pred = pred[:,0].view(-1,1), pred[:,1].view(-1,1), pred[:,2].view(-1,1)
            t21_t, t22_t, c1_t = actual[:,0].view(-1,1), actual[:,1].view(-1,1), actual[:,2].view(-1,1)
            loss = torch.abs((c1_pred*torch.exp(-self.time_new/t21_pred) + (1.0 - c1_pred)*torch.exp(-self.time_new/t22_pred)) - (c1_t*torch.abs(-self.time_new/t21_t) + (1.0 - c1_t)*torch.exp(-self.time_new/t22_t))).mean()
            return loss


    def OneNet(self, pred_param, actual_param, pred_lam, oracle_lam, reg_pred, reg_actual):
        loss = ((1e5*(pred_lam - oracle_lam))**2).mean() + ((1e5*(reg_pred - reg_actual))**2).mean() + (((pred_param-actual_param)*self.scale)**2).mean() 
        return loss









class customLosses():
    def __init__(self, device):
#         super().__init__()
        self.dev = device
        self.weight_lP = torch.tensor([1,1, 1/100], dtype = float).to(device)
        self.weight_hP = torch.tensor([1,1, 1/150], dtype = float).to(device)
    
        
        
    def scaled_customLoss(self, pred, actual):
        diff_matrix = torch.zeros(actual.shape, dtype=float).to(self.dev)
        idxs = actual[:,2] < 0.20 #Where c1_truth < 0.20 penalize highly (self.weight_hP)
        diff_matrix[idxs] = ((pred[idxs] - actual[idxs])/self.weight_hP)**2
        diff_matrix[~idxs]= ((pred[~idxs] - actual[~idxs])/self.weight_lP)**2
        loss = diff_matrix.mean()
        return loss
    
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity = 'selu')
#         nn.init.xavier_uniform(m.weight)

#===============================================================================#

class LambdaSelection_layer(nn.Module):

    def __init__(self, in_channel, out1, out2, k1, k2, fc1, fc2, out_dim):
        super().__init__()
        self.conv1   = nn.Conv1d(in_channel, out1, k1)
        self.conv2   = nn.Conv1d(out1, out2, k2)
#         self.dropout = nn.Dropout(0.5)
        self.fc1   = nn.Linear(31232, fc1) #Manually change -- need to find correct formula to calc
        self.fc2   = nn.Linear(fc1, fc2)
        self.outputMap = nn.Linear(fc2, out_dim)

    def forward(self,x):
        x = F.relu(self.conv1(x)) #set bias to false????????????
        x = F.relu(self.conv2(x))
        x = torch.flatten(x,1) #Flatten all dimenesions except batch
        x = F.relu(self.fc1(x))
#         x = self.dropout(x)
        x = F.relu(self.fc2(x))
#         x = self.dropout(x)
        x = self.outputMap(x)
        return x


class NLLS_layer(nn.Module):

    def __init__(self, NLLS_solver):
        super().__init__()
        self.solver = NLLS_solver

    def forward(self, ND, lam):
        reg = self.solver(ND, lam)
        return reg
    
    def backward(self):
        pass

class ParameterEstimation_layer(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, out_dim):
        super().__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
        self.hidden3   = nn.Linear(h3_dim, h4_dim)
        self.outputMap = nn.Linear(h4_dim, out_dim)

    def forward(self,x):
        x = F.relu(self.inputMap(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.outputMap(x)
        return x




class Lambda_ParameterEstimation_NN(nn.Module):

    def __init__(self, in_channel_LL, out1_LL, out2_LL, k1_LL, k2_LL, fc1_LL, fc2_LL, out_dim_LL, NLLS_solver, in_dim_PE, h1_dim_PE, h2_dim_PE, h3_dim_PE, h4_dim_PE, out_dim_PE):
        super().__init__()
        self.lambdablock = LambdaSelection_layer(in_channel_LL, out1_LL, out2_LL, k1_LL, k2_LL, fc1_LL, fc2_LL, out_dim_LL)
        self.NLLSblock = NLLS_layer(NLLS_solver)
        self.parameterblock = ParameterEstimation_layer(in_dim_PE, h1_dim_PE, h2_dim_PE, h3_dim_PE, h4_dim_PE, out_dim_PE)
    
    def forward(self, ND):
        lam = self.lambdablock(ND)
        #Do we want an activation layer between each? Probably not. 
        reg = self.NLLSblock(ND, lam)
        out = self.parameterblock(reg) # Keep ND???
        return out, lam



class oneLoss():
    def __init__(self, alpha_factor, c1_factor, device):
        self.alpha = alpha_factor
        self.scale = torch.tensor([1.0,1.0,c1_factor], dtype = float, device = device)
    
    def full_loss(self, pred_param, actual_param, pred_lam, actual_lam):
        loss = (((pred_param-actual_param)*self.scale)**2).mean()**0.5 + self.alpha*(((pred_lam - actual_lam)**2).mean()**0.5)
        return loss
#===============================================================================#

class OneNetLamParam(nn.Module):     
    def __init__(self, in_channel, out1, out2, k1, k2, fc1, fc2, out_dim_lam, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, out_dim):
        super().__init__()

        # self.conv1   = nn.Conv1d(in_channel, out1, k1)
        # self.conv2   = nn.Conv1d(out1, out2, k2)
#         self.dropout = nn.Dropout(0.5)
        self.fc1   = nn.Linear(in_channel, fc1) #Manually change -- need to find correct formula to calc
        self.fc2   = nn.Linear(fc1, fc2)
        self.outputLam = nn.Linear(fc2, out_dim_lam)

        self.decode = nn.Linear(out_dim_lam, fc2)
        self.ND_reg = nn.Linear(fc2, in_dim)
        self.start = nn.Linear(in_dim, h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
        self.hidden3   = nn.Linear(h3_dim, h4_dim)
        self.outputMap = nn.Linear(h4_dim, out_dim)

    def forward(self,x):
        # x = F.relu(self.conv1(x)) #set bias to false????????????
        # x = F.relu(self.conv2(x))
        # x = torch.flatten(x,1) #Flatten all dimenesions except batch
        x = F.relu(self.fc1(x))
#         x = self.dropout(x)
        x = F.relu(self.fc2(x))
#         x = self.dropout(x)
        lam = self.outputLam(x)
        x = F.relu(self.decode(lam))
        reg = self.ND_reg(x)
        x = F.relu(self.start(reg))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.outputMap(x)
        return x, lam, reg

#===============================================================================#

class SimpleNN(nn.Module):

    def __init__(self, in_dim, h1_dim, out_dim):
        super().__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim)
        self.outputMap = nn.Linear(h1_dim, out_dim)

    def forward(self,x):
        x = F.relu(self.inputMap(x))
        x = self.outputMap(x)
        return x

#===============================================================================#
class LambdaTraining_FC_NN(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, out_dim):
        super().__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
        self.hidden3   = nn.Linear(h3_dim, h4_dim)
        self.outputMap = nn.Linear(h4_dim, out_dim)

    def forward(self,x):
        x = F.relu(self.inputMap(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.outputMap(x)
        return x

#===============================================================================#


#===============================================================================#
class ILR_x2(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, h5_dim, h6_dim,h7_dim, h8_dim, out_dim, act, norm=False):
        super().__init__()
        self.act = None
        self.norm = norm
        bias = True
        if self.norm:
            bias = False
            _norm_fxn = nn.BatchNorm1d
            self.norm1 = _norm_fxn(h1_dim)
            self.norm2 = _norm_fxn(h2_dim)
            self.norm3 = _norm_fxn(h3_dim)
            self.norm4 = _norm_fxn(h4_dim)
            self.norm5 = _norm_fxn(h5_dim)
            self.norm6 = _norm_fxn(h6_dim)
            self.norm7 = _norm_fxn(h7_dim)
            self.norm8 = _norm_fxn(h8_dim)
            
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'selu':
            self.act = nn.SELU()
            
            
            
        self.inputMap  = nn.Linear(in_dim, h1_dim, bias = bias)
        self.hidden1   = nn.Linear(h1_dim, h2_dim, bias = bias)
        self.hidden2   = nn.Linear(h2_dim, h3_dim, bias = bias)
        self.hidden3   = nn.Linear(h3_dim, h4_dim, bias = bias)
        self.hidden4   = nn.Linear(h4_dim, h5_dim, bias = bias)
        self.hidden5   = nn.Linear(h5_dim, h6_dim, bias = bias)
        self.hidden6   = nn.Linear(h6_dim, h7_dim, bias = bias)
        self.hidden7   = nn.Linear(h7_dim, h8_dim, bias = bias)
        self.outputMap = nn.Linear(h8_dim, out_dim)
           
        
    def forward(self,ND):
        if self.norm:
            x = self.inputMap(ND)
            x = self.norm1(x)
            x = self.act(x)
            
            x = self.hidden1(x)
            x = self.norm2(x)
            x = self.act(x)
            
            x = self.hidden2(x)
            x = self.norm3(x)
            x = self.act(x)
            
            x = self.hidden3(x)
            x = self.norm4(x)
            x = self.act(x)
            
            x = self.hidden4(x)
            x = self.norm5(x)
            x = self.act(x)
            
            x = self.hidden5(x)
            x = self.norm6(x)
            x = self.act(x)
            
            x = self.hidden6(x)
            x = self.norm7(x)
            x = self.act(x)
            
            x = self.hidden7(x)
            x = self.norm8(x)
            x = self.act(x)
            
            x = self.outputMap(x)
        
        else:
            x = self.act(self.inputMap(ND))
            x = self.act(self.hidden1(x))
            x = self.act(self.hidden2(x))
            x = self.act(self.hidden3(x))
            x = self.act(self.hidden4(x))
            x = self.act(self.hidden5(x))
            x = self.act(self.hidden6(x))
            x = self.act(self.hidden7(x))
            x = self.outputMap(x)
        
        return x

#===============================================================================#
class ParamEstimation3(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, h5_dim, h6_dim, out_dim):
        super().__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
        self.hidden3   = nn.Linear(h3_dim, h4_dim)
        self.hidden4   = nn.Linear(h4_dim, h5_dim)
        self.hidden5   = nn.Linear(h5_dim, h6_dim)

        self.outputMap = nn.Linear(h6_dim, out_dim)

    def forward(self,x):
        x = F.relu(self.inputMap(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = self.outputMap(x)
        return x
#===============================================================================#

class LambdaTraining_FC_Simple(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, out_dim):
        super().__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
#         self.hidden3   = nn.Linear(h3_dim, h4_dim)
        self.outputMap = nn.Linear(h3_dim, out_dim)

    def forward(self,x):
        x = F.relu(self.inputMap(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
#         x = F.relu(self.hidden3(x))
        x = self.outputMap(x)
        return x

#===============================================================================#

class LambdaTraining_FC_5(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, h5_dim, out_dim):
        super().__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
        self.hidden3   = nn.Linear(h3_dim, h4_dim)
        self.hidden4   = nn.Linear(h4_dim, h5_dim)
        self.outputMap = nn.Linear(h5_dim, out_dim)

    def forward(self,x):
        x = F.relu(self.inputMap(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.outputMap(x)
        return x

#===============================================================================#


class LambdaTraining_FC_NN_SELU(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, h5_dim, out_dim):
        super(LambdaTraining_FC_NN_SELU, self).__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
        self.hidden3   = nn.Linear(h3_dim, h4_dim)
        self.hidden4   = nn.Linear(h4_dim, h5_dim)
        self.outputMap = nn.Linear(h5_dim, out_dim)

    def forward(self,x):
        x = F.selu(self.inputMap(x))
        x = F.selu(self.hidden1(x))
        x = F.selu(self.hidden2(x))
        x = F.selu(self.hidden3(x))
        x = F.selu(self.hidden4(x))
        x = self.outputMap(x)
        return x

#===============================================================================#

class LambdaTraining_FC_NN_SELU_BatchNorm(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, h5_dim, out_dim):
        super(LambdaTraining_FC_NN_SELU_BatchNorm, self).__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim, bias = False)
        self.inputMap_bn = nn.BatchNorm1d(h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim, bias = False)
        self.hidden1_bn = nn.BatchNorm1d(h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim, bias = False)
        self.hidden2_bn = nn.BatchNorm1d(h3_dim)
        self.hidden3   = nn.Linear(h3_dim, h4_dim, bias = False)
        self.hidden3_bn = nn.BatchNorm1d(h4_dim)
        self.hidden4   = nn.Linear(h4_dim, h5_dim, bias = False)
        self.hidden4_bn = nn.BatchNorm1d(h5_dim)
        self.outputMap = nn.Linear(h5_dim, out_dim)

    def forward(self,x):
        x = F.selu(self.inputMap_bn(self.inputMap(x)))
        x = F.selu(self.hidden1_bn(self.hidden1(x)))
        x = F.selu(self.hidden2_bn(self.hidden2(x)))
        x = F.selu(self.hidden3_bn(self.hidden3(x)))
        x = F.selu(self.hidden4_bn(self.hidden4(x)))
        x = self.outputMap(x)
        return x

#===============================================================================#



class LambdaTraining_FC_NN_SELU_BatchNorm_OneLayer(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, h5_dim, out_dim):
        super(LambdaTraining_FC_NN_SELU_BatchNorm_OneLayer, self).__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self._bn = nn.BatchNorm1d(h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
        
        self.hidden3   = nn.Linear(h3_dim, h4_dim)
        self.hidden4   = nn.Linear(h4_dim, h5_dim)
        self.outputMap = nn.Linear(h5_dim, out_dim)

    def forward(self,x):
        x = F.selu(self.inputMap(x))
        x = F.selu(self.hidden1(x))
        x = F.selu(self.hidden2(self._bn(x)))
        x = F.selu(self.hidden3(x))
        x = F.selu(self.hidden4(x))
        x = self.outputMap(x)
        return x

#===============================================================================#


class LambdaTraining_FC_NN_SELU_BatchNorm_PreInput(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, h5_dim, out_dim):
        super(LambdaTraining_FC_NN_SELU_BatchNorm_PreInput, self).__init__()
        self.inputMap_bn = nn.BatchNorm1d(in_dim)
        self.inputMap  = nn.Linear(in_dim, h1_dim, bias = False)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
        self.hidden3   = nn.Linear(h3_dim, h4_dim)
        self.hidden4   = nn.Linear(h4_dim, h5_dim)
        self.outputMap = nn.Linear(h5_dim, out_dim)

    def forward(self,x):
        x = F.selu(self.inputMap(self.inputMap_bn(x)))
        x = F.selu(self.hidden1(x))
        x = F.selu(self.hidden2(x))
        x = F.selu(self.hidden3(x))
        x = F.selu(self.hidden4(x))
        x = self.outputMap(x)
        return x

#===============================================================================#


class LambdaTraining_FC_NN_SELU_TanH(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, h4_dim, h5_dim, out_dim):
        super(LambdaTraining_FC_NN_SELU_TanH, self).__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
        self.hidden3   = nn.Linear(h3_dim, h4_dim)
        self.hidden4   = nn.Linear(h4_dim, h5_dim)
        self.outputMap = nn.Linear(h5_dim, out_dim)

    def forward(self,x):
        x = F.tanh(self.inputMap(x))
        x = F.selu(self.hidden1(x))
        x = F.selu(self.hidden2(x))
        x = F.selu(self.hidden3(x))
        x = F.selu(self.hidden4(x))
        x = self.outputMap(x)
        return x

#===============================================================================#


class LambdaTraining_FC_NN_SELU_H3(nn.Module):

    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, out_dim):
        super(LambdaTraining_FC_NN_SELU_H3, self).__init__()
        self.inputMap  = nn.Linear(in_dim, h1_dim)
        self.hidden1   = nn.Linear(h1_dim, h2_dim)
        self.hidden2   = nn.Linear(h2_dim, h3_dim)
        self.outputMap = nn.Linear(h3_dim, out_dim)

    def forward(self,x):
        x = F.selu(self.inputMap(x))
        x = F.selu(self.hidden1(x))
        x = F.selu(self.hidden2(x))
        x = self.outputMap(x)
        return x

#===============================================================================#
class LambdaTraining_FC_NN_Convolutional_SELU_32TE(nn.Module):

    def __init__(self, in_channel, out1, out2, k1, k2, fc1, fc2, out_dim):
        super().__init__()
        self.conv1   = nn.Conv1d(in_channel, out1, k1)
        self.conv2   = nn.Conv1d(out1, out2, k2)
#         self.dropout = nn.Dropout(0.5)
        self.fc1   = nn.Linear(6656, fc1) #Manually change -- need to find correct formula to calc
        self.fc2   = nn.Linear(fc1, fc2)
        self.outputMap = nn.Linear(fc2, out_dim)

    def forward(self,x):
        x = F.relu(self.conv1(x)) #set bias to false????????????
        x = F.relu(self.conv2(x))
        x = torch.flatten(x,1) #Flatten all dimenesions except batch
        x = F.relu(self.fc1(x))
#         x = self.dropout(x)
        x = F.relu(self.fc2(x))
#         x = self.dropout(x)
        x = self.outputMap(x)
        return x

#===============================================================================#


class LambdaTraining_FC_NN_Convolutional_SELU(nn.Module):

    def __init__(self, in_channel, out1, out2, k1, k2, fc1, fc2, out_dim):
        super().__init__()
        self.conv1   = nn.Conv1d(in_channel, out1, k1, padding=2)
        self.conv2   = nn.Conv1d(out1, out2, k2, padding=1)
#         self.dropout = nn.Dropout(0.5)
        self.fc1   = nn.Linear(8192, fc1) #Manually change -- need to find correct formula to calc
        self.fc2   = nn.Linear(fc1, fc2)
        self.outputMap = nn.Linear(fc2, out_dim)

    def forward(self,x):
        x = F.relu(self.conv1(x)) #set bias to false????????????
        # print("conv1", x.shape)
        x = F.relu(self.conv2(x))
        # print("conv2", x.shape)
        x = torch.flatten(x,1) #Flatten all dimenesions except batch
        x = F.relu(self.fc1(x))
#         x = self.dropout(x)
        x = F.relu(self.fc2(x))
#         x = self.dropout(x)
        x = self.outputMap(x)
        return x

#===============================================================================#


class LambdaTraining_Shashank(nn.Module):

    def __init__(self, in_dim, in_channel, out1, out2, k1, k2, fc1, fc2, out_dim):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, in_dim//2)
        self.conv1   = nn.Conv1d(in_channel, out1, k1)
        self.conv2   = nn.Conv1d(out1, out2, k2)
#         self.dropout = nn.Dropout(0.5)
        self.fc1   = nn.Linear(2560, fc1) #Manually change -- need to find correct formula to calc
        self.fc2   = nn.Linear(fc1, fc2)
        self.outputMap = nn.Linear(fc2, out_dim)

    def forward(self,x):
        x = F.relu(self.fc_in(x))
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x)) #set bias to false????????????
        x = F.relu(self.conv2(x))
        x = torch.flatten(x,1) #Flatten all dimenesions except batch
        x = F.relu(self.fc1(x))
#         x = self.dropout(x)
        x = F.relu(self.fc2(x))
#         x = self.dropout(x)
        x = self.outputMap(x)
        return x



#===============================================================================#

class LambdaTraining_FC_NN_Convolutional_SELU_Deep(nn.Module):

    def __init__(self, in_channel, out1, out2, k1, k2, fc1, fc2, fc4, fc5, fc6, out_dim):
        super().__init__()
        self.conv1   = nn.Conv1d(in_channel, out1, k1)
        self.conv2   = nn.Conv1d(out1, out2 , k2) # Change out2 --> out1??? See Note 1. 
        #IMPLEMENT RESNET Add another CONV BLOCK!!!
#         self.conv3 = nn.Conv1d(out2, out2??? , k2???) --> Note 1. see what parts have to be kept same for res to be added
        
        self.dropout = nn.Dropout(0.50)
        
#         n_size = self._get_conv_output
        self.fc1   = nn.Linear(58, fc1) #Manually change -- need to find correct formula to calc
        self.fc2   = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(65536, 512)
        self.fc4 = nn.Linear(512, fc4)
        self.fc5 = nn.Linear(fc4, fc5)
        self.fc6 = nn.Linear(fc5, fc6)
        
        
        self.outputMap = nn.Linear(fc6, out_dim)

    def forward(self,x):
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = F.selu(self.fc1(x))
#         x = self.dropout(x)
        x = F.selu(self.fc2(x))
#         x = self.dropout(x)
        x = torch.flatten(x,1) #Flatten all dimenesions except batch
        x = F.selu(self.fc3(x))
#         x = self.dropout(x)
        x = F.selu(self.fc4(x))
#         x = self.dropout(x)
        x = F.selu(self.fc5(x))
        x = F.selu(self.fc6(x))
        x = self.dropout(x)
        x = self.outputMap(x)
        return x

#===============================================================================#

#===============================================================================#

def conv3x(in_planes, out_planes, stride = 1, groups =1, dilation = 1):
    return nn.Conv1d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = dilation,
                     groups = groups, bias = False, dilation = dilation)

def conv1x(in_planes, out_planes, stride = 1):
    return nn.Conv1d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)


class bottleNeck_Block(nn.Module):
    
    expansion = 1 #Was 4?
    
    def __init__(self, inplanes, planes, stride=1, downsample = None, groups = 1, base_width = 64, dilation = 1, norm_layer = nn.BatchNorm1d):
        super().__init__()

        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv1x(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x(width, width, stride = stride, groups = groups, dilation = dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x(width, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace= True)
        self.downsample = downsample

        self.stride = stride
    
    
    def forward(self,x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
#             print("Downsampling...")
            identity = self.downsample(x)
#             print("IDENTITY: ", identity.shape)
        
#         print("OUT: ", out.shape)
        out += identity
        out = self.relu(out)

        return out


class ResNET_2PE(nn.Module):

    def __init__(self, in_channels, layers, out_dim, groups = 1, width_per_group = 64, zero_init_residual = True):
        super().__init__()
        
        norm_layer  = nn.BatchNorm1d
        self._norm_layer = norm_layer 
        
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.dilation = 1
        self.block = bottleNeck_Block
        
        self.conv1 = nn.Conv1d(in_channels, self.inplanes, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(self.block, 64, layers[0])
        self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, layers[3], stride=2)
        
        self.fc = nn.Linear(512 * self.block.expansion*self.block.expansion * 2, out_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, bottleNeck_Block):
                    nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
        
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x(self.inplanes, planes * block.expansion, stride = stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride = stride, downsample = downsample, groups = self.groups,
                            base_width = self.base_width, dilation = previous_dilation, norm_layer = norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation = self.dilation, norm_layer = norm_layer))

        return nn.Sequential(*layers)
        

    def forward(self,x):
        # See note [TorchScript super()]
#         print("Start")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
    #         x = self.maxpool(x)
#         print("Start L1")
        
        x = self.layer1(x)
#         print("Start L2")
        
        x = self.layer2(x)
#         print("Start L3")
        
        x = self.layer3(x)
        x = self.layer4(x)

    #         x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

#===============================================================================#
def resnet152(**kwargs):
    model = ResNET_2PE(layers = [3, 8, 36, 3], **kwargs)
    # if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model

def resnet18(**kwargs):
    model = ResNET_2PE(layers = [2, 2, 2, 2], **kwargs)
    # if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model

# def resnet152(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-152 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet152', [3, 8, 36, 3], **kwargs)


#===============================================================================#

def fc_noBias(in_dim, out_dim):
    return nn.Linear(in_dim, out_dim, bias = False)

class fc_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = fc_noBias(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        
        self.fc2 = fc_noBias(out_dim, out_dim*4)
        self.bn2 = nn.BatchNorm1d(out_dim*4)
        
        self.fc3 = fc_noBias(out_dim*4, in_dim)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.relu = nn.ReLU(inplace=True)
        
        
        
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.bn3(out)
        
#         if self.downsample is not None:
#             identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class FC_Deep_ResNetLike(nn.Module):
    
    def __init__(self, layers, input_size = 32, final_out = 1):
        super().__init__()
        self.expansion = 2
        self.block = fc_block
        norm_layer  = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.in_1 = input_size*self.expansion
                
        self.fc1 = nn.Linear(input_size, self.in_1, bias = False)
        self.bn1 = norm_layer(self.in_1)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(self.block, 64, layers[0])
        self.layer2 = self._make_layer(self.block, 128, layers[1])
        self.layer3 = self._make_layer(self.block, 256, layers[2])
        self.layer4 = self._make_layer(self.block, 512, layers[3])
        
        self.fc_final = nn.Linear(self.in_1, final_out)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        
        for m in self.modules():
            if isinstance(m, bottleNeck_Block):
                nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
# sizes = 
    def _make_layer(self, block, out_dim, blocks):

        layers = []
        
        for _ in range(0, blocks):
            layers.append(block(self.in_1, out_dim))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fc_final(x)

        return x
    
def myFC_resnet50(**kwargs):
    model = FC_Deep_ResNetLike(layers = [3, 4, 6, 3], **kwargs)
    # if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
    return model


# TRANSFORMERS ===========================================================

# import torch
# import torch.nn as nn
# from timm.models.layers import DropPath, to_2tuple
# from timm.models.registry import register_model
# from timm.models.vision_transformer import Mlp, PatchEmbed

# class ModifiedDeiTBlock(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads, add_bias_kv=qkv_bias, dropout=attn_drop)
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x), x, x)[0])
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

# class ModifiedDeiT(nn.Module):
#     def __init__(self, sequence_length = 64, patch_size=1, in_chans=1, num_classes=3, embed_dim=768, depth=12, num_heads=12,
#                  mlp_ratio=4.0, qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.0, drop_path_rate=0.01):
#         super().__init__()


#         self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length, embed_dim))

#         self.dropout = nn.Dropout(p=drop_rate)

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.blocks = nn.ModuleList([
#             ModifiedDeiTBlock(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
#             for i in range(depth)])

#         self.norm = nn.LayerNorm(embed_dim)

#         self.head = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
# #         x = self.patch_embed(x)
# #         x = x.flatten(2).transpose(1, 2)

# #         cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
# #         x = torch.cat((cls_tokens, x), dim=1)
#         x = x.unsqueeze(-1) + self.pos_embed
#         x = self.dropout(x)

#         for block in self.blocks:
#             x = block(x)

#         x = self.norm(x)
#         x = self.head(x[:,0])
#         return x



# class VisionTransformer(nn.Module):
#     def __init__(self, input_dim, num_classes, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm, **kwargs):
#         super().__init__()

# #         self.num_features  = self.embed_dim = embed_dim
# #         self.patch_size = patch_size
# #         self.patch_dim = (input_dim // patch_size) ** 2
        
# #         self.patch_embedding = nn.Linear(patch_size**2, embed_dim)
# #         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embedding = nn.Parameter(torch.zeros(1, input_dim, embed_dim))
#         self.dropout = nn.Dropout(kwargs.get('dropout_rate', 0.1))

#         dpr = [x.item() for x in torch.linspace(0, kwargs.get('dropout_rate', 0.1), depth)]  # stochastic depth decay rule

#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
#                 norm_layer=norm_layer, dropout_rate=dpr[i], **kwargs)
#             for i in range(depth)])

#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#         self.fc = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         # x shape: (batch_size, num_patches, patch_dim)
#         batch_size = x.shape[0]
# #         display(x.shape)

#         # Add classification token and position embedding
#         x = x.unsqueeze(-1) + self.pos_embedding
#         x = self.dropout(x)

#         # Transformer blocks
#         for block in self.blocks:
#             x = block(x)

#         # Extract classification token and pass through fully connected layer
# #         cls_token = x[:, 0]
#         x = self.norm(x)
#         out = self.fc(x[:,0])
#         return out


# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=None, dropout_rate=0.):
#         super().__init__()

#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, dropout_rate=dropout_rate)
#         self.drop_path = DropPath(dropout_rate)
#         self.norm2 = norm_layer(dim)
#         self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout_rate=dropout_rate)

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=True, dropout_rate=0.):
#         super().__init__()

#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(dropout_rate)
#         self.proj = nn.Linear(dim, dim)
    
#     def forward(self, x):
#         # x shape: (batch_size, num_patches + 1, embed_dim)
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.attn_drop(x)
#         return x


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features, dropout_rate=0.):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden_features, in_features)
#         self.drop = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# class DropPath(nn.Module):
#     def __init__(self, drop_prob=0.):
#         super().__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         if self.training:
#             keep_prob = 1 - self.drop_prob
#             mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep_prob
#             x = torch.where(mask, x, torch.zeros_like(x))
#         return x


