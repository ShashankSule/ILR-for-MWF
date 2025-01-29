from cvxopt import solvers, matrix, spdiag, exp, mul

from makeSignals import myTrueModel_2param, myTrueModel

import numpy as np

from scipy.optimize import least_squares, curve_fit

import torch

from torch.utils.data import Dataset

from typing import List
import config
from scipy import special

solvers.options['show_progress'] = False







#-----------------------------------#

##    Wrappers for NLLS Solvers    ##

#-----------------------------------#



# A) CVXOpt: L^2 Regularized NLLS

#--------------------------------

def cvxopt_l2RegularizedNLLS(G, d, times, ld, D, p_0, 

                     *,signalType):

    """

    Approximates the solution of the L^2 regularized least-squares problem:



    (*)        argmin_{p in R^3} (||G(p) - d||_2^2 + (ld**2)*||D*p||_2^2)



    where G : R^3 -> R^(num_times) == an operator that maps the triple of

    parameters (c, T21, T22) to a discrete signal measured at the input times.

    

    Input:

    ------

        1. G () - Function that describes a clean signal. For us, myTrueModel.

        2. d (Numpy array of length len(times)) - noisy signal

        3. times (numpy array) - times at which the signal == measured.

        4. ld (nonnegative float) - regularization parameter.

        5. D (Numpy array or CVXOpt matrix of size (3,3)) - Diagonal matrix of

            weights in the penalty term.

        6. p_0 (Array of length 3) - Initial guess of solution for the NLLS solver. 

        7. signalType (string, optional kwarg) - Type of signal. Choices are

            biexponential, power, quadratic, or sinusoidal.



    Output:

    -------

        1. p (Array of length 3) - The parameter estimates, i.e., the approximate

            solution to (*).

    """

    model_dim = 3 # number of parameters input to the model G

    t = matrix(times)

    def F(p=None, z=None):

        if p == None: return 0, matrix(p_0, (model_dim,1))

        if min(p) <= 0.0: return None, None

        # 1. Residual: G(p) - d

        res = matrix(G(times, p[0], p[1], p[2], signalType=signalType)) - d

        # 2. objective to minimize: model error + penalty

        f = sum(res**2) + (ld**2)*sum((mul(D,p))**2)

        # 3. Partial derivatives of signal model G w.r.t. parameters

        pd_c1_G  = exp(-t/p[1]) - exp(-t/p[2])   # pd G w.r.t. c1

        pd_T21_G = mul(p[0]*t/(p[1]**2),         # pd G w.r.t. T2,1

                       exp(-t/p[1]))   

        pd_T22_G = mul((1.0 - p[0])*t/(p[2]**2), # pd G w.r.t. T2,2

                       exp(-t/p[2]))

        # 3. Partial derivatives of objective f w.r.t. parameters

        Df_1 = 2.0*( sum(mul(res, pd_c1_G)) +   # pd model error w.r.t. c1

                     ((ld*D[0])**2)*p[0] )    # pd penalty w.r.t. c1 

        Df_2 = 2.0*( sum(mul(res, pd_T21_G)) +  # pd model error w.r.t. T2,1

                     ((ld*D[1])**2)*p[1] )    # pd penalty w.r.t. T2,1

        Df_3 = 2.0*( sum(mul(res, pd_T22_G)) +  # pd model error w.r.t. T2,2

                     ((ld*D[2])**2)*p[2] )    # pd penalty w.r.t. T2,2

        # 4. Assemble gradient of objective transposed.: (grad f)^T (a row vector)

        Df = matrix([ Df_1, Df_2, Df_3 ], (1,3))

        if z == None: return f, Df

        # 5. 2nd order partial derivatives of signal model G w.r.t. parameters

        pd2_c1_G  = 0.0                          # 2nd pd G w.r.t. c1

        pd2_T21_G = mul(-2.0/p[1] + t/(p[1]**2), # 2nd pd G w.r.t. T2,1

                        pd_T21_G) 

        pd2_T22_G = mul(-2.0/p[2] + t/(p[2]**2), # 2nd pd G w.r.t. T2,2

                        pd_T22_G)

        pd_T21_c1_G = mul(t/(p[1]**2), exp(-t/p[1]))  # pd G w.r.t. T2,1 and c1

        pd_T22_c1_G = -mul(t/(p[2]**2), exp(-t/p[2])) # pd G w.r.t. T2,2 and c1

        pd_T21_T22_g = 0.0                          # pd G w.r.t. T2,1 and T2,2

        # 6. Components of the Hessian H of the objective f

        H_11 = 2.0*z[0]*( sum(mul(res, pd_c1_G**2)) # 2nd pd model error w.r.t. c1

                          + (D[0]*ld)**2 )        # 2nd pd penalty w.r.t. c1

        H_21 = 2.0*z[0]*sum(mul(res,                # pd model error w.r.t. c1 and T2,1

                                 mul(pd_T21_G, pd_c1_G) + pd_T21_c1_G))

        H_31 = 2.0*z[0]*sum(mul(res,                # pd model error w.r.t. c1 and T2,2

                                 mul(pd_T22_G, pd_c1_G) + pd_T22_c1_G))

        H_22 = 2.0*z[0]*( sum(mul(res,              # 2nd pd model error w.r.t. T2,1

                                  pd_T21_G**2 + pd2_T21_G)) 

                          + (D[1]*ld)**2 )        # 2nd pd penalty w.r.t. T2,1

        H_32 = 2.0*z[0]*sum(mul(res,                # pd model error w.r.t. T2,1 and T2,2

                                mul(pd_T22_G, pd_T21_G))) 

        H_33 = 2.0*z[0]*( sum( mul(res,             # 2nd pd model error w.r.t. T2,2

                                  pd_T22_G**2 + pd2_T22_G))

                         + (D[2]*ld)**2 )         # 2nd pd penalty w.r.t. T2,2

        # 7. Assemble Hessian of objective. CVXOpt only needs lower triangle...

        #    ...due to symmetry.

        H = matrix([H_11, H_21, H_31,

                    0.0, H_22, H_32,

                    0.0, 0.0, H_33], (3,3) )

        return f, Df, H

    soln = solvers.cp(F)['x']

#    print(f'The solution is: \n\t\t{soln}')

    return soln





# A.2) CVXOpt: L^2 Regularized NLLS Two Parameters

# -----------------------------------------------

def cvxopt_l2RegularizedNLLS_Two_Parameters(G, d, times, ld, D, p_0,

                             *, signalType):

    """

    Approximates the solution of the L^2 regularized least-squares problem:



    (*)        argmin_{p in R^3} (||G(p) - d||_2^2 + (ld**2)*||D*p||_2^2)



    where G : R^3 -> R^(num_times) == an operator that maps the triple of

    parameters (c, T21, T22) to a discrete signal measured at the input times.



    Input:

    ------

        1. G () - Function that describes a clean signal. For us, myTrueModel.

        2. d (Numpy array of length len(times)) - noisy signal

        3. times (numpy array) - times at which the signal == measured.

        4. ld (nonnegative float) - regularization parameter.

        5. D (Numpy array or CVXOpt matrix of size (3,3)) - Diagonal matrix of

            weights in the penalty term.

        6. p_0 (Array of length 3) - Initial guess of solution for the NLLS solver.

        7. signalType (string, optional kwarg) - Type of signal. Choices are

            biexponential, power, quadratic, or sinusoidal.



    Output:

    -------

        1. p (Array of length 3) - The parameter estimates, i.e., the approximate

            solution to (*).

    """

    model_dim = 2  # number of parameters input to the model G

    t = matrix(times)



    def F(p=None, z=None):

        if p == None: return 0, matrix(p_0, (model_dim, 1))

        if min(p) <= 0.0: return None, None

        # 1. Residual: G(p) - d

        res = matrix(G(times, p[0], p[1], signalType=signalType)) - d

        # 2. objective to minimize: model error + penalty

        f = sum(res ** 2) + (ld ** 2) * sum((mul(D, p)) ** 2)

        # 3. Partial derivatives of signal model G w.r.t. parameters

#        pd_c1_G = exp(-t / p[1]) - exp(-t / p[2])  # pd G w.r.t. c1

        pd_T21_G = mul(0.6 * t / (p[0] ** 2),  # pd G w.r.t. T2,1

                       exp(-t / p[0]))

        pd_T22_G = mul((1.0 - 0.6) * t / (p[1] ** 2),  # pd G w.r.t. T2,2

                       exp(-t / p[1]))

        # 3. Partial derivatives of objective f w.r.t. parameters

#        Df_1 = 2.0 * (sum(mul(res, pd_c1_G)) +  # pd model error w.r.t. c1

#                      ((ld * D[0]) ** 2) * p[0])  # pd penalty w.r.t. c1

        Df_2 = 2.0 * (sum(mul(res, pd_T21_G)) +  # pd model error w.r.t. T2,1

                      ((ld * D[0]) ** 2) * p[0])  # pd penalty w.r.t. T2,1

        Df_3 = 2.0 * (sum(mul(res, pd_T22_G)) +  # pd model error w.r.t. T2,2

                      ((ld * D[1]) ** 2) * p[1])  # pd penalty w.r.t. T2,2

        # 4. Assemble gradient of objective transposed.: (grad f)^T (a row vector)

        Df = matrix([Df_2, Df_3], (1, 2))

        if z == None: return f, Df

        # 5. 2nd order partial derivatives of signal model G w.r.t. parameters

#        pd2_c1_G = 0.0  # 2nd pd G w.r.t. c1

        pd2_T21_G = mul(-2.0 / p[0] + t / (p[0] ** 2),  # 2nd pd G w.r.t. T2,1

                        pd_T21_G)

        pd2_T22_G = mul(-2.0 / p[1] + t / (p[1] ** 2),  # 2nd pd G w.r.t. T2,2

                        pd_T22_G)

        pd_T21_c1_G = mul(t / (p[0] ** 2), exp(-t / p[0]))  # pd G w.r.t. T2,1 and c1

        pd_T22_c1_G = -mul(t / (p[1] ** 2), exp(-t / p[1]))  # pd G w.r.t. T2,2 and c1

        pd_T21_T22_g = 0.0  # pd G w.r.t. T2,1 and T2,2

        # 6. Components of the Hessian H of the objective f

        H_11 = 2.0 * z[0] * (sum(mul(res, 0.6 ** 2))  # 2nd pd model error w.r.t. c1

                             + (0.6 * ld) ** 2)  # 2nd pd penalty w.r.t. c1

        H_21 = 2.0 * z[0] * sum(mul(res,  # pd model error w.r.t. c1 and T2,1

                                    mul(pd_T21_G, 0.6) + pd_T21_c1_G))

        H_31 = 2.0 * z[0] * sum(mul(res,  # pd model error w.r.t. c1 and T2,2

                                    mul(pd_T22_G, 0.6) + pd_T22_c1_G))

        H_22 = 2.0 * z[0] * (sum(mul(res,  # 2nd pd model error w.r.t. T2,1

                                     pd_T21_G ** 2 + pd2_T21_G))

                             + (D[0] * ld) ** 2)  # 2nd pd penalty w.r.t. T2,1

        H_32 = 2.0 * z[0] * sum(mul(res,  # pd model error w.r.t. T2,1 and T2,2

                                    mul(pd_T22_G, pd_T21_G)))

        H_33 = 2.0 * z[0] * (sum(mul(res,  # 2nd pd model error w.r.t. T2,2

                                     pd_T22_G ** 2 + pd2_T22_G))

                             + (D[1] * ld) ** 2)  # 2nd pd penalty w.r.t. T2,2

        # 7. Assemble Hessian of objective. CVXOpt only needs lower triangle...

        #    ...due to symmetry.

        H = matrix([H_11, H_21, H_31,

                    0.0, H_22, H_32,

                    0.0, 0.0, H_33], (3, 3))

        return f, Df, H



    soln = solvers.cp(F)['x']

    #    print(f'The solution is: \n\t\t{soln}')

    return soln







##-------------------------------------##

##   SciPy: Wrapper for least_squares  ##

##-------------------------------------##



# B.1) Nonregularized NLLS - 2 parameters

# ---------------------------------------

def least_squares_2param(G, d, times, c1, c2, p_0,

               *, signalType="biexponential"):

    """

    Approximates the solution of the nonlinear least-squares problem:



    (*)        argmin_{p in R^2} ||G(p) - d||_2^2



    where G : R^2 -> R^(num_times) == an operator that maps the pair of

    parameters (T21, T22) to a discrete signal measured at the input times.



    Input:

    ------

        1. G (callable) - Function that describes a clean signal.

            For us, myTrueModel.

        2. d (Numpy array of length len(times)) - noisy signal

        3. times (numpy array) - times at which the signal == measured.

        4. c1 (float) - initial fraction of component one (associated with T21) - T21 == smaller and decays faster

        5. c2 (float) - initial fraction of component two (associated with T22) - T21 == larger and decays slower

        6. p_0 (Array of length 2) - Initial guess of solution for the NLLS solver.

        7. signalType (string, optional kwarg) - Type of signal. Choices are

            biexponential, power, quadratic, or sinusoidal.



    Output:

    -------

        1. p (Array of length 2) - The parameter estimates, i.e., the approximate

            solution to (*).

    """



    def residuals(p):

        """ The vector G(p) - d in R^(number of times) of model residuals """

        res = G(times, c1, c2, p[0], p[1], signalType=signalType) - d

        # the residuals input to least_squares *must* be a Numpy array

        if type(res) == torch.Tensor:

            res = res.numpy()

            return np.sqrt(2.0) * res[0, :]

        elif type(res) == np.ndarray:

            return np.sqrt(2.0) * res



    #        print(f'\tThe vector of model residuals is: {res}')

    #        print(f'\tThe size of the residual is: {np.size(res)}')

    #        print(f'\tThe penalty is: {pen}')

    #        print(f'\tThe size of the penalty is: {np.size(pen)}')



    # B) Bounds on the parameters

    lb_T21, ub_T21 = 0.0, np.inf  # lb can be small & positive to enforce nonnegativity

    lb_T22, ub_T22 = 0.0, np.inf



    # Computes the least squares solution. Uses 2 point finite-differences...

    #  ...to approximate the gradient. Could also use 3 point or take exact gradient.

    out = least_squares(residuals, p_0,

                        bounds=([lb_T21, lb_T22], [ub_T21, ub_T22]))



    return out.x


def least_squares_in_parallel(G, d, times, c1, c2, p_0,

               *, signalType="biexponential"):

    """

    Approximates the solution of the nonlinear least-squares problem:



    (*)        argmin_{p in R^2} ||G(p) - d||_2^2



    where G : R^2 -> R^(num_times) == an operator that maps the pair of

    parameters (T21, T22) to a discrete signal measured at the input times.



    Input:

    ------

        1. G (callable) - Function that describes a clean signal.

            For us, myTrueModel.

        2. d (Numpy array of length len(times)) - noisy signal

        3. times (numpy array) - times at which the signal == measured.

        4. c1 (float) - initial fraction of component one (associated with T21) - T21 == smaller and decays faster

        5. c2 (float) - initial fraction of component two (associated with T22) - T21 == larger and decays slower

        6. p_0 (Array of length 2) - Initial guess of solution for the NLLS solver.

        7. signalType (string, optional kwarg) - Type of signal. Choices are

            biexponential, power, quadratic, or sinusoidal.



    Output:

    -------

        1. p (Array of length 2) - The parameter estimates, i.e., the approximate

            solution to (*).

    """



    def residuals(p):

        """

        The vector G(p) - d in R^(number of times) of model residuals

        """
        print('p', p)
        print('G', G)
        res = G(times, c1, c2, p[0], p[1], signalType=signalType) - d

        # the residuals input to least_squares *must* be a Numpy array

        if type(res) == torch.Tensor:

            res = res.numpy()

            return np.sqrt(2.0) * res[0, :]

        elif type(res) == np.ndarray:

            return np.sqrt(2.0) * res



    #        print(f'\tThe vector of model residuals is: {res}')

    #        print(f'\tThe size of the residual is: {np.size(res)}')

    #        print(f'\tThe penalty is: {pen}')

    #        print(f'\tThe size of the penalty is: {np.size(pen)}')



    # B) Bounds on the parameters

    lb_T21, ub_T21 = 0.0, np.inf  # lb can be small & positive to enforce nonnegativity

    lb_T22, ub_T22 = 0.0, np.inf



    # Computes the least squares solution. Uses 2 point finite-differences...

    #  ...to approximate the gradient. Could also use 3 point or take exact gradient.

    out = least_squares(residuals, p_0,

                        bounds=([lb_T21, lb_T22], [ub_T21, ub_T22]))



    return out.x





# B.2) Nonregularized NLLS - 3 parameters

#----------------------------------------

def least_squares_3param(G, d, times, p_0, 

                         *,signalType="biexponential"):

    """

    Approximates the solution of the nonlinear least-squares problem:



    (*)        argmin_{p in R^3} ||G(p) - d||_2^2



    where G : R^3 -> R^(num_times) == an operator that maps the triple of

    parameters (c, T21, T22) to a discrete signal measured at the input times.

    

    Input:

    ------

        1. G (callable) - Function that describes a clean signal. 

            For us, myTrueModel.

        2. d (Numpy array of length len(times)) - noisy signal

        3. times (numpy array) - times at which the signal == measured.

        6. p_0 (Array of length 3) - Initial guess of solution for the NLLS solver. 

        7. signalType (string, optional kwarg) - Type of signal. Choices are

            biexponential, power, quadratic, or sinusoidal.



    Output:

    -------

        1. p (Array of length 3) - The parameter estimates, i.e., the approximate

            solution to (*).

    """

    

    def residuals(p):

        """ 

        The vector G(p) - d in R^(number of times) of model residuals

        """

        res = G(times, p[0], p[1], p[2], signalType=signalType) - d

        # the residuals input to least_squares *must* be a Numpy array

        if type(res) == torch.Tensor:

            res = res.numpy()

            return np.sqrt(2.0)*res[0,:]

        elif type(res) == np.ndarray:

            return np.sqrt(2.0)*res

#        print(f'\tThe vector of model residuals is: {res}')

#        print(f'\tThe size of the residual is: {np.size(res)}')

#        print(f'\tThe penalty is: {pen}')

#        print(f'\tThe size of the penalty is: {np.size(pen)}')





    # B) Bounds on the parameters

    lb_c1, ub_c1 = 0.0, 1.0

    lb_T21, ub_T21 = 0.0, np.inf # lb can be small & positive to enforce nonnegativity

    lb_T22, ub_T22 = 0.0, np.inf



    # Computes the least squares solution. Uses 2 point finite-differences... 

    #  ...to approximate the gradient. Could also use 3 point or take exact gradient.

    out = least_squares(residuals, p_0,

                        bounds=([lb_c1, lb_T21, lb_T22], [ub_c1, ub_T21, ub_T22]))



    return out.x











# B.3) L^2 Regularized NLLS - 2 parameters

#-----------------------------------------

def least_squares_l2Regularized_2param(G, d, times, ld, D, c1, c2, p_0,

                                       *, signalType="biexponential"):

    """

    Approximates the solution of the L^2 regularized least-squares problem:



    (*)        argmin_{p in R^2} (||G(p,c) - d||_2^2 + (ld**2)*||D*p||_2^2)



    where G : R^4 -> R^(num_times) == an operator that maps the four

    parameters (c1, c2, T21, T22) to a discrete signal measured at the input times.



    Input:

    ------

        1. G (callable) - Function that describes a clean signal.

            For us, myTrueModel.

        2. d (Numpy array of length len(times)) - noisy signal

        3. times (numpy array) - times at which the signal == measured.

        4. ld (nonnegative float) - regularization parameter.

        5. D (Numpy array) - Diagonal matrix of

            weights in the penalty term.

        6. c1 (float) - initial fraction of component one (associated with T21) - T21 == smaller and decays faster

        7. c2 (float) - initial fraction of component two (associated with T22) - T21 == larger and decays slower

        8. p_0 (Array of length 3) - Initial guess of solution for the NLLS solver.

        9. signalType (string, optional kwarg) - Type of signal. Choices are

            biexponential, power, quadratic, or sinusoidal.



    Output:

    -------

        1. p (Array of length 2) - The parameter estimates, i.e., the approximate

            solution to (*).

    """

    def pen_vec(p):

        """

        The vector D*p in R^2.

        """

        return np.multiply(D, p)



    def res_vec(p):

        """

        The vector G(p) - d in R^(number of times) of model residuals concatenated

        with the penalty vector ld*D*p.

        """

        res = G(times, c1, c2, p[0], p[1], signalType=signalType) - d

        pen = ld * pen_vec(p)

        # the residuals input to least_squares *must* be a Numpy array

        if type(res) == torch.Tensor:

            res = res.numpy()

            return np.sqrt(2.0) * np.concatenate((res[0, :], pen))

        elif type(res) == np.ndarray:

            # print(f'\tThe size of the residual is: {np.size(res)}')

            # print(f'\tThe size of the penalty is: {np.size(pen)}')

            # print(f'\tThe size of the vector == {np.size(np.concatenate((res,pen)))}')

            return np.sqrt(2.0) * np.concatenate((res, pen))



        

    # B) Bounds on the parameters

    lb_T21, ub_T21 = 0.0, np.inf  # lb can be small & positive to enforce nonnegativity

    lb_T22, ub_T22 = 0.0, np.inf



    # Computes the least squares solution. Uses 2 point finite-differences...

    #  ...to approximate the gradient. Could also use 3 point or take exact gradient.

    out = least_squares(res_vec, p_0,

                        bounds=([lb_T21, lb_T22], [ub_T21, ub_T22]))



    return out.x









# B.4) L^2 regularized NLLS - 3 parameters

#-----------------------------------------

def least_squares_l2Regularized_3param(G, d, times, ld, D, p_0, 

                                       *,signalType="biexponential"):

    """

    Approximates the solution of the L^2 regularized least-squares problem:



    (*)        argmin_{p in R^3} (||G(p) - d||_2^2 + (ld**2)*||D*p||_2^2)



    where G : R^3 -> R^(num_times) == an operator that maps the triple of

    parameters (c, T21, T22) to a discrete signal measured at the input times.

    

    Input:

    ------

        1. G (callable) - Function that describes a clean signal. 

            For us, myTrueModel.

        2. d (Numpy array of length len(times)) - noisy signal

        3. times (numpy array) - times at which the signal == measured.

        4. ld (nonnegative float) - regularization parameter.

        5. D (Numpy array) - Diagonal matrix of

            weights in the penalty term.

        6. p_0 (Array of length 3) - Initial guess of solution for the NLLS solver. 

        7. signalType (string, optional kwarg) - Type of signal. Choices are

            biexponential, power, quadratic, or sinusoidal.



    Output:

    -------

        1. p (Array of length 3) - The parameter estimates, i.e., the approximate

            solution to (*).

    """



    def pen_vec(p):

        """

        The vector D*p in R^3.

        """

        return np.multiply(D,p)

    

    def res_vec(p):

        """ 

        The vector G(p) - d in R^(number of times) of model residuals concatenated 

        with the penalty vector ld*D*p.

        """

        res = G(times, p[0], p[1], p[2], signalType=signalType) - d

        pen = ld*pen_vec(p)

        # the residuals input to least_squares *must* be a Numpy array

        if type(res) == torch.Tensor:

            res = res.numpy()

            return np.sqrt(2.0)*np.concatenate((res[0,:], pen))

        elif type(res) == np.ndarray:

            return np.sqrt(2.0)*np.concatenate((res, pen))

#        print(f'\tThe vector of model residuals is: {res}')

#        print(f'\tThe size of the residual is: {np.size(res)}')

#        print(f'\tThe penalty is: {pen}')

#        print(f'\tThe size of the penalty is: {np.size(pen)}')





    # B) Bounds on the parameters

    lb_c1, ub_c1 = 0.0, 1.0

    lb_T21, ub_T21 = 0.0, np.inf # lb can be small & positive to enforce nonnegativity

    lb_T22, ub_T22 = 0.0, np.inf



    # Computes the least squares solution. Uses 2 point finite-differences... 

    #  ...to approximate the gradient. Could also use 3 point or take exact gradient.

    out = least_squares(res_vec, p_0,

                        bounds=([lb_c1, lb_T21, lb_T22], [ub_c1, ub_T21, ub_T22]))



    return out.x











##---------------------------------##

##   SciPy: Wrapper for curve_fit  ##

##---------------------------------##



##  C.1) Nonregularized NLLS - 2 parameters

##-----------------------------------------

def curve_fit_2param(d, times, c1, c2, p_0,

                     *, signalType="biexponential",

                     lb_T21=0.0, lb_T22=0.0, ub_T21=np.inf, ub_T22=np.inf):

    """

    Approximates the solution of the nonlinear least-squares problem:



    (*)        argmin_{p in R^2} ||G(p) - d||_2^2



    where G : R^2 -> R^(num_times) == an operator that maps the pair of

    parameters (T21, T22) to a discrete signal measured at the input times.



    Input:

    ------

        1. d (Numpy array of length len(times)) - noisy signal

        2. times (numpy array) - times at which the signal == measured.

        3. c1 (float) - initial fraction of component one (associated with T21) - T21 == smaller and decays faster

        4. c2 (float) - initial fraction of component two (associated with T22) - T21 == larger and decays slower

        5. p_0 (Array of length 2) - Initial guess of solution for the NLLS solver.

        6. signalType (string, optional kwarg) - Type of signal. Choices are

            biexponential, power, quadratic, or sinusoidal.

        7. lb_T21 (float) - Lower T21 bound - set to 0.0

        8. lb_T22 (float) - Lower T22 bound - set to 0.0

        9. ub_T21 (float) - Upper T21 bound - set to inf

        10. ub_T22 (float) - Upper T22 bound - set to inf



    Output:

    -------

        1. p (Array of length 2) - The parameter estimates, i.e., the approximate

            solution to (*).

    """



    # A) Define curve to fit data to. 

    #    Needs to have a call signature (xdata, parameter

    def signal(xdata, p1, p2):



        if signalType == "biexponential":



            # 1. number of signal acquisition times

            num_times = len(xdata) - 2



            # 2. extract data

            t_vec = xdata[0:num_times]

            fast_frac = xdata[num_times]

            slow_frac = xdata[num_times+1]



            # 3. compute discrete signal and return

            return fast_frac*np.exp(-t_vec/p1) + slow_frac*np.exp(-t_vec/p2)



    # B) Bounds on the parameters

    #lb_T21, ub_T21 = 0.0, np.inf  # lb can be small & positive to enforce nonnegativity

    #lb_T22, ub_T22 = 0.0, np.inf



    # C) Fit given noisy signal d to curve and constants

    # Uses 2 point finite-differences to approximate the gradient.

    # Could also use 3 point or take exact gradient.

    t_dim = times.ndim

    indep_var = np.concatenate((times,

                                np.array(c1,ndmin=t_dim), np.array(c2,ndmin=t_dim)))

                 


    try:
        #print('function inputs are')
        #print('     signal:     ', signal)
        #print('     indep_var:     ', indep_var)
        #print('     d:     ', d)
        #print('     p_0:     ', p_0)
        #print('     bounds:     ', ([lb_T21, lb_T22], [ub_T21, ub_T22]))
        opt_val = curve_fit(signal, indep_var, d,  # curve, xdata, ydata

                            p0=p_0,  # initial guess

                            bounds=([lb_T21, lb_T22], [ub_T21, ub_T22]),

                            method='trf', maxfev=5000)
    except RuntimeError:
        print("maximum number of function evaluations == exceeded")
        try:
            opt_val = curve_fit(signal, indep_var, d,  # curve, xdata, ydata

                                p0=p_0/2,  # initial guess

                                bounds=([lb_T21, lb_T22], [ub_T21, ub_T22]),

                                method='trf', maxfev=5000)
        except RuntimeError:
            print("maximum number of function evaluations == exceeded")
            opt_val = p_0

        opt_val[0] = p_0



    return opt_val[0]



def curve_fit_3param(d, times, p_0,

                     *, signalType="biexponential",

                     lb_T21=0.0, lb_T22=0.0, lb_c1=0.0, ub_T21=np.inf, ub_T22=np.inf, ub_c1=np.inf):

    """

    Approximates the solution of the nonlinear least-squares problem:



    (*)        argmin_{p in R^2} ||G(p) - d||_2^2



    where G : R^2 -> R^(num_times) == an operator that maps the pair of

    parameters (T21, T22) to a discrete signal measured at the input times.



    Input:

    ------

        1. d (Numpy array of length len(times)) - noisy signal

        2. times (numpy array) - times at which the signal == measured.

        3. p_0 (Array of length 3) - Initial guess of solution for the NLLS solver.

        4. signalType (string, optional kwarg) - Type of signal. Choices are

            biexponential, power, quadratic, or sinusoidal.

        5. lb_T21 (float) - Lower T21 bound - set to 0.0

        6. lb_T22 (float) - Lower T22 bound - set to 0.0
        
        7. lb_c1 (float) - Lower c1 bound - set to 0.0

        7. ub_T21 (float) - Upper T21 bound - set to inf

        8. ub_T22 (float) - Upper T22 bound - set to inf
        
        9. ub_c1 (float) - Upper c1 bound - set to inf



    Output:

    -------

        1. p (Array of length 3) - The parameter estimates, i.e., the approximate

            solution to (*).

    """



    # A) Define curve to fit data to. 

    #    Needs to have a call signature (xdata, parameter

    def signal(xdata, p1, p2, p3):



        if signalType == "biexponential":



            # 1. number of signal acquisition times

            num_times = len(xdata)



            # 2. extract data

            t_vec = xdata[0:num_times]



            # 3. compute discrete signal and return

            return p3*np.exp(-t_vec/p1) + (1.0-p3)*np.exp(-t_vec/p2)



    # B) Bounds on the parameters

    #lb_T21, ub_T21 = 0.0, np.inf  # lb can be small & positive to enforce nonnegativity

    #lb_T22, ub_T22 = 0.0, np.inf



    # C) Fit given noisy signal d to curve and constants

    # Uses 2 point finite-differences to approximate the gradient.

    # Could also use 3 point or take exact gradient.

    t_dim = times.ndim

    indep_var = times

                 


    try:
        #print('function inputs are')
        #print('     signal:     ', signal)
        #print('     indep_var:     ', indep_var)
        #print('     d:     ', d)
        #print('     p_0:     ', p_0)
        #print('     bounds:     ', ([lb_T21, lb_T22, lb_c1], [ub_T21, ub_T22, ub_c1]))
        opt_val = curve_fit(signal, indep_var, d,  # curve, xdata, ydata

                            p0=p_0,  # initial guess

                            bounds=([lb_T21, lb_T22, lb_c1], [ub_T21, ub_T22, ub_c1]),

                            method='trf', maxfev=5000)
    except RuntimeError:
        print("maximum number of function evaluations == exceeded")
        try:
            opt_val = curve_fit(signal, indep_var, d,  # curve, xdata, ydata

                                p0=p_0/2,  # initial guess

                                bounds=([lb_T21, lb_T22, lb_c1], [ub_T21, ub_T22, ub_c1]),

                                method='trf', maxfev=5000)
        except RuntimeError:
            print("maximum number of function evaluations == exceeded")
            opt_val = p_0

        opt_val[0] = p_0



    return opt_val[0]








##  C) L2-regularized NLLS - 2 parameters

##---------------------------------------

def curve_fit_l2Regularized_2param(d, times, ld, D, c1, c2, p_0,

                                   *, signalType="biexponential",

                                   lb_T21=0.0, lb_T22=0.0, ub_T21=np.inf, ub_T22=np.inf):

    """

    Approximates the solution of the L^2 regularized least-squares problem:



    (*)        argmin_{p in R^2} (||G(p,c) - d||_2^2 + (ld**2)*||D*p||_2^2)



    where G : R^2 -> R^(num_times) == an operator that maps the pair of

    parameters (T21, T22) to a discrete signal measured at the input times.



    Input:

    ------

        1. d (Numpy array of length len(times)) - noisy signal

        2. times (numpy array) - times at which the signal == measured

        3. ld (float) - regularization parameter

        4. c1 (float) - initial fraction of component one (associated with T21) - T21 == smaller and decays faster

        5. c2 (float) - initial fraction of component two (associated with T22) - T21 == larger and decays slower

        6. p_0 (Array of length 2) - Initial guess of solution for the NLLS solver

        7. signalType (string, optional kwarg) - Type of signal. Choices are

            biexponential, power, quadratic, or sinusoidal.

        7. lb_T21 (float) - Lower T21 bound - set to 0.0

        8. lb_T22 (float) - Lower T22 bound - set to 0.0

        9. ub_T21 (float) - Upper T21 bound - set to inf

        10. ub_T22 (float) - Upper T22 bound - set to inf



    Output:

    -------

        1. p (Array of length 2) - The parameter estimates, i.e., the approximate

            solution to (*).

    """



    # A) Define curve to fit data to. 

    #    Needs to have a call signature (xdata, parameter

    def signal(xdata, p1, p2):



        if signalType == "biexponential":



            # 1. number of signal acquisition times

            num_times = len(xdata) - 3



            # 2. extract data

            t_vec = xdata[0:num_times]   # times

            fast_frac = xdata[num_times]   # fraction of fast component

            slow_frac = xdata[num_times+1] # fraction of slow component

            reg_param = xdata[num_times+2] # regularization parameter



            # 3. calculate penalty

            params = np.array([p1, p2], dtype=np.float64)

            penalty_vec = ld*np.multiply(D,params)



            # 4. concatenate and return

            return np.concatenate((fast_frac*np.exp(-t_vec/p1) + slow_frac*np.exp(-t_vec/p2),

                                  penalty_vec))

                                  



    # B) Bounds on the parameters

    #lb_T21, ub_T21 = 0.0, np.inf  # lb can be small & positive to enforce nonnegativity

    #lb_T22, ub_T22 = 0.0, np.inf



    # C) Fit given dependent variable to curve and independent variable

    # Uses 2 point finite-differences to approximate the gradient.

    # Could also use 3 point or take exact gradient.

    t_dim = times.ndim

    indep_var = np.concatenate((times,

                                np.array(c1,ndmin=t_dim), np.array(c2,ndmin=t_dim),

                                np.array(ld,ndmin=t_dim)))



    d_dim = d.ndim

    depen_var = np.concatenate((d, np.array(0.0, ndmin=d_dim), np.array(0.0,ndmin=d_dim)))


    try:
        opt_val = curve_fit(signal, indep_var, depen_var,  # curve, xdata, ydata

                            p0=p_0,  # initial guess

                            bounds=([lb_T21, lb_T22], [ub_T21, ub_T22]),

                            method="trf",

                            max_nfev=500)
    except RuntimeError:
        print("maximum number of function evaluations == exceeded")





    # returns estimate. second index in estimated covariance matrix

    return opt_val[0]





def curve_fit_l2Regularized_3param(d, times, ld, D, p_0,

                                   *, signalType="biexponential",

                                   lb_T21=0.0, lb_T22=0.0, lb_c1=0.0, ub_T21=np.inf, ub_T22=np.inf, ub_c1=np.inf):

    """

    Approximates the solution of the L^2 regularized least-squares problem:



    (*)        argmin_{p in R^2} (||G(p,c) - d||_2^2 + (ld**2)*||D*p||_2^2)



    where G : R^2 -> R^(num_times) == an operator that maps the pair of

    parameters (T21, T22) to a discrete signal measured at the input times.



    Input:

    ------

        1. d (Numpy array of length len(times)) - noisy signal

        2. times (numpy array) - times at which the signal == measured

        3. ld (float) - regularization parameter

        4. p_0 (Array of length 3) - Initial guess of solution for the NLLS solver

        5. signalType (string, optional kwarg) - Type of signal. Choices are

            biexponential, power, quadratic, or sinusoidal.

        6. lb_T21 (float) - Lower T21 bound - set to 0.0

        7. lb_T22 (float) - Lower T22 bound - set to 0.0

        8. ub_T21 (float) - Upper T21 bound - set to inf

        9. ub_T22 (float) - Upper T22 bound - set to inf



    Output:

    -------

        1. p (Array of length 2) - The parameter estimates, i.e., the approximate

            solution to (*).

    """



    # A) Define curve to fit data to. 

    #    Needs to have a call signature (xdata, parameter


    def signal(t_vec, p1, p2, p3):

#         p1 = t1
#         p2 = t2
#         p3 = c1
        

        if signalType == "biexponential":
            

            return p3*np.exp(-t_vec/p1) + (1.0-p3)*np.exp(-t_vec/p2)


    def expectation_Rice(xdata,t21,t22, c1):
        sigma=np.max(d)/config.SNR
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

    return opt_val[0]






def make_l2_trajectory(times, noisy_signal,

                       reg_params, c1_target, c2_target, p_0, D=None,

                       solver="scipy_least_squares",

                       lb_T21=0.0, lb_T22=0.0, ub_T21=np.inf, ub_T22=np.inf):

    """

    Generates L^2 regularization trajectories from a noisy signal measured at

    the times times noisy_signal and a collection of regularization parameters

    reg_params. Let d:=noisy_signal. Then an L^2 regularized nonlinear

    least squares (NLLS) problem



    (*)        argmin_{p in R^3} (||G(p) - d||_2^2 + (ld^2)*||D*p||_2^2)



    == solved for each parameter ld in reg_params.

    

    Input:

    ------

        1. times (Numpy array) - The acquisition times for the noisy signal.

        2. noisy_signal (Numpy array of length len(times)) - A noisy signal, e.g.,

            biexponential decay.

        3. reg_params (Array of length traj_length) - Collection of regularization

               parameters for the NLLS problem.

        4. p_0 (Numpy array of length 3) - Initial guess for the NLLS solver. First 

            component == c1, 2nd == T21, and third == T22.

        5. D (Numpy array of length 3) - Array of weights for the penalty term.

        6. solver (str) - The method used to solve the LS problem. Either 

            "cvxopt" or "scipy_least_squares" or "scipy_curve_fit".

        7. lb_T21 (float) - Lower T21 bound - set to 0.0

        8. lb_T22 (float) - Lower T22 bound - set to 0.0

        9. ub_T21 (float) - Upper T21 bound - set to inf

        10. ub_T22 (float) - Upper T22 bound - set to inf



    Output:

    -------

        1. c1_traj (Array of length traj_length) - Regularization trajectory

               for the initial fraction of component 1, c1.

        2. T21_traj (Array of length traj_length) - Regularization trajectory

               for the fast time constant, T21.

        3. T22_traj (Array of length traj_length) - Regularization trajectory

               for the slow time constant, T22.       

    """

    if len(p_0) == 2: num_params = 2

    elif len(p_0) == 3: num_params = 3



    # A) Initialize

    #--------------

    # num_times = len(noisy_signal); traj_length = len(reg_params)

    if np.isscalar(reg_params):
        traj_length = 1
        reg_params = torch.tensor([reg_params])
    else:
        traj_length = len(reg_params)

    T21_traj = torch.empty(traj_length)  # T21 regularization trajectory

    T22_traj = torch.empty(traj_length)  # T22 regularization trajectory





    # B) Make trajectories

    #---------------------

    if num_params == 2:

        if solver == "cvxopt":

            d = matrix(noisy_signal)

            if D == None:

                D = matrix([1.0, 1.0, 1.0])

            else:

                D = matrix(D)



            for lbda, k in zip(reg_params, range(0, traj_length)):

                #            print(f"\t\t\t-- The current lambda and iteration are {lbda} and {k+1}")

                if type(lbda) == torch.Tensor: lbda = lbda.numpy()

                T21_traj[k], T22_traj[k] = cvxopt_l2RegularizedNLLS_Two_Parameters(myTrueModel_2param,

                                                                                       noisy_signal,

                                                                                       times,

                                                                                       lbda,

                                                                                       D,

                                                                                       c1_target,

                                                                                       c2_target,

                                                                                       p_0,

                                                                                       signalType="biexponential")

            return T21_traj, T22_traj



        elif solver == "scipy_least_squares":

            for lbda, k in zip(reg_params, range(0, traj_length)):

                #            print(f"\t\t\t-- The current lambda and iteration are {lbda} and {k+1}")

                if type(lbda) == torch.Tensor: lbda = lbda.numpy()

                T21_traj[k], T22_traj[k] = least_squares_l2Regularized_2param(myTrueModel_2param,

                                                                              noisy_signal,

                                                                              times,

                                                                              lbda,

                                                                              D,

                                                                              c1_target,

                                                                              c2_target,

                                                                              p_0,

                                                                              signalType="biexponential")

            return T21_traj, T22_traj



        elif solver == "scipy_curve_fit":

            for lbda, k in zip(reg_params, range(0, traj_length)):

                #            print(f"\t\t\t-- The current lambda and iteration are {lbda} and {k+1}")

                if type(lbda) == torch.Tensor: lbda = lbda.numpy()

                T21_traj[k], T22_traj[k] = curve_fit_l2Regularized_2param(noisy_signal,

                                                                          times,

                                                                          lbda,

                                                                          D,

                                                                          c1_target,

                                                                          c2_target,

                                                                          p_0,

                                                                          signalType="biexponential",

                                                                          lb_T21=lb_T21,

                                                                          lb_T22=lb_T22,

                                                                          ub_T21=ub_T21,

                                                                          ub_T22=ub_T22)

            return T21_traj, T22_traj



    if num_params == 3:

        c1_traj  = torch.empty(traj_length)  # c1 regularization trajectory



        if solver == "cvxopt":

            d = matrix(noisy_signal)

            if D == None:

                D = matrix([1.0, 1.0, 1.0])

            else:

                D = matrix(D)



            for lbda, k in zip(reg_params, range(0, traj_length)):

                #            print(f"\t\t\t-- The current lambda and iteration are {lbda} and {k+1}")

                if type(lbda) == torch.Tensor: lbda = lbda.numpy()

                c1_traj[k], T21_traj[k], T22_traj[k] = cvxopt_l2RegularizedNLLS(myTrueModel,

                                                                                    d,

                                                                                    times,

                                                                                    lbda,

                                                                                    D,

                                                                                    p_0,

                                                                                    signalType="biexponential")

            return T21_traj, T22_traj, c1_traj



        elif solver == "scipy_least_squares":

            for lbda, k in zip(reg_params, range(0, traj_length)):

                #            print(f"\t\t\t-- The current lambda and iteration are {lbda} and {k+1}")

                if type(lbda) == torch.Tensor: lbda = lbda.numpy()

                T21_traj[k], T22_traj[k], c1_traj[k] = least_squares_l2Regularized_3param(myTrueModel,

                                                                                          noisy_signal,

                                                                                          times,

                                                                                          lbda,

                                                                                          D,

                                                                                          p_0,

                                                                                          signalType="biexponential")

            return c1_traj, T21_traj, T22_traj



        elif solver == "scipy_curve_fit":

            # if (not type(c1_target) == torch.Tensor) and (not np.isscalar(c1_target)):
            #     new_signal = noisy_signal[k]
            #     c1_t = c1_target[k]
            #     c2_t = c2_target[k]
            # else:
            #     new_signal = noisy_signal
            #     c1_t = c1_target
            #     c2_t = c2_target


            for lbda, k in zip(reg_params, range(0, traj_length)):

                #            print(f"\t\t\t-- The current lambda and iteration are {lbda} and {k+1}")


                # if (not type(c1_target) == torch.Tensor) and (not np.isscalar(c1_target)):
                #     new_signal = noisy_signal[k]
                #     c1_t = c1_target[k]
                #     c2_t = c2_target[k]
                #     t21_t = p_0.iloc[k][0]
                #     t22_t = p_0.iloc[k][1]
                #     p_0_new = np.array([t21_t, t22_t])
#                     print(p_0.iloc(k))
#                     print(new_signal.shape, c1_t, c2_t, t21_t, t22_t, p_0_new)
#                     assert False
                    
                    
                # else:
                new_signal = noisy_signal
                c1_t = c1_target
                c2_t = c2_target
                p_0_new = p_0                    


                if type(lbda) == torch.Tensor: lbda = lbda.detach().numpy()

                T21_traj[k], T22_traj[k], c1_traj[k] = curve_fit_l2Regularized_3param(new_signal,

                                                                                      times,

                                                                                      lbda,

                                                                                      D,

                                                                                      p_0_new,

                                                                                      signalType="biexponential")

            return c1_traj, T21_traj, T22_traj









