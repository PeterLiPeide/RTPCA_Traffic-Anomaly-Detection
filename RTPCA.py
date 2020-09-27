"""
This is a package to perform robust tensor PCA decomposition and anomaly detection

This module includes functions for optimizing the objective function using ADMM

Author: Peide (Peter) Li

Used packages: numpy, tensorly
"""
#%%

import os
import numpy as np 
import tensorly as tl 

from numpy.linalg import norm
import progressbar
from time import sleep

import TrafficData_Generator as TF_anomaly


from sklearn.ensemble import IsolationForest
from sklearn.covariance import MinCovDet
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt 

np.random.seed(999)

def soft_threshold(X, sigma):
    """
    Performing a soft-thresholding. It is the solution for vector lasso

    Since our algorithm takes L1 norm on all elements in a tensor, it is indeed a vectorized form of lasso
    and thus X can be a np array with any shape
    X: a tensor input
    sigma: a scalar
    
    return: a thresholded tensor
    """
    tmp = abs(X) - sigma
    tmp[np.where(tmp < 0)] = 0
    return tmp * np.sign(X)

def soft_moden(T, tau, n):
    '''
    Tensor mode n SVD truncation
    input: 
    tensor T
    turncate parameter: tau
    mode index: n

    return thresholded matrix and the correpsonding updated unclear norm
    '''
    modeN = tl.unfold(T, n)
    U, S, Vt = np.linalg.svd(modeN, full_matrices=True)
    S = S - tau
    S_index = np.where(S > 0)[0][-1]
    Unew = U[:, 0 : S_index + 1]
    VtNew = Vt[0 : S_index + 1, :]
    return np.dot(Unew, np.dot(np.diag(S[0 : S_index + 1]), VtNew))


def soft_HOSVD(TX, TV, lambda1, psi, F3 = None):
    """
    update ancelliary tenosrs W^{i} 
    """
    Nmode = len(TX.shape)
    Tshape = TX.shape
    res_W = []
    for i in range(Nmode):
        if not F3:
            tmpW = soft_moden(TX - TV, lambda1 * psi[i], i)
        else:
            tmpW = soft_moden(TX - TV + F3[i], lambda1 * psi[i], i)
        res_W.append(tl.fold(tmpW, i, Tshape))
    return res_W
        





class Robust_Tensor_PCA():

    """
    Object of robust tensor principle componenet analysis

    The object saves all regularization parameters as well as tensor data
    Methods of this objects includes model estimation, prediction and evaluation
    """

    def __init__(self, TensorX, parameter_list = None):
        """
        Initial the object by putting raw traffic tensor. 
        Parameter_list is optional. If not provided, the function can initial the parameter itself.

        Object contains three tensor components:
        1. Low rank part: self.W
        2. Sparse part: self.V
        3. Original traffic volumn: self.X

        Ancixilluary variables are also initialized at the beginning.
        """
        self.X = TensorX
        self.W = np.zeros(self.X.shape)
        self.V = np.zeros(self.X.shape)
        self.totaldim = np.prod(self.X.shape)

        # Auxillaury variables
        self.Wlist = [np.zeros(self.W.shape) for i in range(len(self.X.shape))]
        self.F1, self.F2 = np.zeros(self.X.shape), np.zeros(self.X.shape)
        self.S, self.T = np.zeros(self.X.shape), np.zeros(self.X.shape)
        self.Dmatrix = np.eye(self.X.shape[0])
        for i in range(self.X.shape[0] - 1):
            self.Dmatrix[i, i + 1] = -1
        self.Dmatrix[-1, 0] = -1

        self.anomaliscores = None
        self.anomaliscores_MAD = None
        self.anomaliscores_ISOF = None
        self.anomaliscores_SVM = None

        if not parameter_list:
            self.parameter = {'m' : 1 / (np.std(self.X.flatten())), 'lambda1' : 1, 'beta1' : 1 / max(self.X.shape), 'beta2' : 1 / 10 / max(self.X.shape) , 'phi' : [0.01, 1, 10, 0.001], 'rho1' : 1 / (5 * np.std(self.X.flatten())), 'rho2': 1 / (5 * np.std(self.X.flatten()))}
        else:
            self.parameter = parameter_list
        return

    def get_Objective(self):

        """Calculate the Objective function"""
        
        ans = 0
        for i in range(len(self.X.shape)):
            ans += self.parameter['phi'][i] * self.parameter['lambda1'] *norm(tl.unfold(self.Wlist[i], i), 'nuc') + (1 / self.parameter['m']) * norm(tl.unfold(self.X - self.Wlist[i] - self.V, i))

        # Augmented part is calculated seperately. 
        augment_part1 = 0.5 * self.parameter['rho1'] * norm(self.V - self.T + self.F1)
        augment_part2 = 0.5 * self.parameter['rho2'] * norm(tl.fold(np.dot(self.Dmatrix, tl.unfold(self.T, 0)), 0, self.T.shape) - self.S + self.F2)

        # Combine the result for final objective function
        ans += self.parameter['beta1'] * norm(self.V.reshape(self.totaldim), 1) + self.parameter['beta2'] * norm(self.S.reshape(self.totaldim), 1) +  augment_part1 + augment_part2 
        return ans

    def fit(self, gamma1, gamma2, eta = 0.01, maxitor = 50):
        """
        Fiiting anomaly detection models
        (This function estimates the low rank and sparse part in tensor)

        In general, this function also works as an estimator for robust tensor PCA

        optional input:
        eta: average difference (between each iteration, converge)
        maxitor: maximum number of iteration

        gamma1 and gamma2 are learning rates for two augmented lagrangian multipliers

        This function also records the objective function value in fly
        """
        print("Start fitting robust tensor PCA model...\n")
        self.Objective_value = [self.get_Objective()]

        threshold_lambda = self.parameter['m'] * self.parameter['lambda1']
        threshold_V = 2 * self.parameter['m'] * self.parameter['beta1'] / (6 + self.parameter['m'] * self.parameter['rho1'])
        threshold_S = 2 * self.parameter['beta2'] / self.parameter['rho2']

        
        current_eta, current_itor = 1000, 0
        
        # bar = progressbar.ProgressBar(maxval=maxitor, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        # bar.start()
        
        while current_eta > eta and current_itor < maxitor:
            
            prev_obj = self.get_Objective()
            
            self.Wlist = soft_HOSVD(self.X, self.V, threshold_lambda, self.parameter['phi'])
            # Update W
            self.W = sum(self.Wlist) / len(self.Wlist)

            # Update V
            tensor_Tau = 2 * sum([self.X - i for i in self.Wlist]) + self.parameter['m'] * self.parameter['rho1'] * (self.T - self.F1)
            tensor_Tau /= (6 + self.parameter['m'] * self.parameter['rho1'])
            self.V = soft_threshold(tensor_Tau, threshold_V)

            # Update S
            tensor_T = tl.fold(np.dot(self.Dmatrix, tl.unfold(self.T, 0)), 0, self.X.shape) + self.F2
            self.S = soft_threshold(tensor_T, threshold_S)

            # Update T using gradient descent
            tmp_inv = np.linalg.pinv(self.parameter['rho1'] * np.eye(self.T.shape[0]) + self.parameter['rho2'] * np.dot(self.Dmatrix.T, self.Dmatrix))
            tmp_right = self.parameter['rho1'] * tl.unfold(self.V + self.F1, 0) + self.parameter['rho2'] * np.dot(self.Dmatrix.T, tl.unfold(self.S - self.F2, 0))
            self.T = tl.fold(np.dot(tmp_inv, tmp_right), 0, self.T.shape)


            # Update Dual parameters
            self.F1 += gamma1 * (self.V - self.T)
            self.F2 += gamma2 * (tl.fold(np.dot(self.Dmatrix, tl.unfold(self.T, 0)), 0, self.T.shape) - self.S)

            current_obj = self.get_Objective()
            current_eta = np.log(prev_obj - current_obj)
            
            self.Objective_value.append(current_obj)
            # bar.update(current_itor + 1)
            current_itor += 1
        # bar.finish()

        return
    
    def plot_history(self, filename):
        """Plot the value of objective function as model is fitted
        
        The model has to be fitted first
        """
        plt.figure(figsize=(12, 9))
        plt.plot(self.Objective_value)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title('Objective Function Values')
        # plt.savefig(filename)
        plt.show()
        return 

    def Find_Anomalies(self, threshold = 700):
        """Find Anomalies by calculating mahalanobis distance
        
        Robust covariance matrix is estimated by Minimum determinant Covaraince (robust estimate)
        Mahalanobis distance is calculated with the robust covariance matrix
        First threshold points are selected as anomalies with largest mahalanobis distance

        Note: This function only works with three modes tensor

        Input:
        threshold: how many potential anomalies are considered; default is 7000
        """

        if self.anomaliscores is None:
            # if anomaly scores are not available, compute anomaly scores first
            self.anomaliscores = np.zeros(self.X.shape)
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[-1]):
                    tmp_data = np.squeeze(self.V[i, :, j]).reshape(-1, 1)
                    try:
                        tmp_Cov = MinCovDet().fit(tmp_data)                     #reshape the squeezed vector so that it contains only one features
                        # Calculate sqrt root of mahalanobis and save as anomalies socres
                        self.anomaliscores[i, :, j] = tmp_Cov.mahalanobis(tmp_data)
                    except ValueError:
                        # If the fiber is all zero, 
                        tmp_Cov = MinCovDet().fit(tmp_data + np.random.normal(0, 0.01, (self.X.shape[1], 1)))
                        self.anomaliscores[i, :, j] = tmp_Cov.mahalanobis(tmp_data)
        anomalies = np.unravel_index(np.argsort(-1 * self.anomaliscores, axis = None)[: threshold], self.anomaliscores.shape)
        return anomalies

    def Find_Anomalies_MAD(self, threshold = 700):
        """
        Find anomalies using MAD score. More robust to non-gaussian distribution

        The MAD based anomaly score definition can be found at Rousseeuw-Anomaly
        """
        if self.anomaliscores_MAD is None:
            # if anomaly scores are not available, compute anomaly scores first
            self.anomaliscores_MAD = np.zeros(self.X.shape)
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[-1]):
                    tmp_data = np.squeeze(self.V[i, :, j]).reshape(-1, 1)
                    med = np.median(tmp_data)
                    MAD = 1.4826 * np.median(np.abs(tmp_data - med))
                    try:
                        self.anomaliscores_MAD[i, :, j] = tmp_data - med / MAD
                    except ValueError:
                        self.anomaliscores_MAD[i, :, j] = np.zeros(self.X.shape[1])
        anomalies = np.unravel_index(np.argsort(-1 * self.anomaliscores_MAD, axis = None)[: threshold], self.anomaliscores_MAD.shape)
        return anomalies

    def Find_Anomalies_ISOForest(self, threshold = 700):
        """
        Find anomalies using MAD score. More robust to non-gaussian distribution

        The MAD based anomaly score definition can be found at Rousseeuw-Anomaly
        """
        if self.anomaliscores_ISOF is None:
            # if anomaly scores are not available, compute anomaly scores first
            self.anomaliscores_ISOF = np.zeros(self.X.shape)
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[-1]):
                    tmp_data = np.squeeze(self.V[i, :, j]).reshape(-1, 1)
                    isomodel = IsolationForest().fit(tmp_data)
                    self.anomaliscores_ISOF[i, :, j] = isomodel.score_samples(tmp_data)
        anomalies = np.unravel_index(np.argsort(self.anomaliscores_ISOF, axis = None)[: threshold], self.anomaliscores_ISOF.shape)
        return anomalies
    
    def Find_Anomalies_SVM(self, threshold = 700):
        """
        Find anomalies using MAD score. More robust to non-gaussian distribution

        The MAD based anomaly score definition can be found at Rousseeuw-Anomaly
        """
        if self.anomaliscores_SVM is None:
            # if anomaly scores are not available, compute anomaly scores first
            self.anomaliscores_SVM = np.zeros(self.X.shape)
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[-1]):
                    tmp_data = np.squeeze(self.V[i, :, j]).reshape(-1, 1)
                    svc = OneClassSVM().fit(tmp_data)
                    self.anomaliscores_SVM[i, :, j] = svc.score_samples(tmp_data)
        anomalies = np.unravel_index(np.argsort(self.anomaliscores_SVM, axis = None)[: threshold], self.anomaliscores_SVM.shape)
        return anomalies


    



class Strong_Constrain_PCA():

    """
    Robust Tensor Decomposition with Strong constrain
    """

    def __init__(self, TensorX, parameter_list = None):
        """
        Initial the object by putting raw traffic tensor. 
        Parameter_list is optional. If not provided, the function can initial the parameter itself.

        Object contains three tensor components:
        1. Low rank part: self.W
        2. Sparse part: self.V
        3. Original traffic volumn: self.X

        Ancixilluary variables are also initialized at the beginning.
        """
        self.X = TensorX
        self.W = np.zeros(self.X.shape)
        self.V = np.zeros(self.X.shape)
        self.totaldim = np.prod(self.X.shape)

        # Auxillaury variables
        self.Wlist = [np.zeros(self.W.shape) for i in range(len(self.X.shape))]
        self.F1, self.F2= np.zeros(self.X.shape), np.zeros(self.X.shape)
        self.F3 = [np.zeros(self.X.shape) for i in range(len(self.X.shape))]
        self.S, self.T = np.zeros(self.X.shape), np.zeros(self.X.shape)
        self.Dmatrix = np.eye(self.X.shape[0])
        for i in range(self.X.shape[0] - 1):
            self.Dmatrix[i, i + 1] = -1
        self.Dmatrix[-1, 0] = -1

        self.anomaliscores = None

        if not parameter_list:
            self.parameter = {'rho3' : 1 / (5 * np.std(self.X.flatten())), 'lambda1' : 1, 'beta1' : 1 / max(self.X.shape), 'beta2' : 1 / 10 / max(self.X.shape) , 'phi' : [0.01, 10, 0.001], 'rho1' : 1 / (5 * np.std(self.X.flatten())), 'rho2': 1 / (5 * np.std(self.X.flatten()))}
        else:
            self.parameter = parameter_list
        return

    def get_Objective(self):

        """Calculate the Objective function"""
        
        ans = 0
        for i in range(len(self.X.shape)):
            ans += self.parameter['phi'][i] * self.parameter['lambda1'] *norm(tl.unfold(self.Wlist[i], i), 'nuc') + (self.parameter['rho3'] / 2) * norm(tl.unfold(self.X - self.Wlist[i] - self.V + self.F3[i], i))

        # Augmented part is calculated seperately. 
        augment_part1 = 0.5 * self.parameter['rho1'] * norm(self.V - self.T + self.F1)
        augment_part2 = 0.5 * self.parameter['rho2'] * norm(tl.fold(np.dot(self.Dmatrix, tl.unfold(self.T, 0)), 0, self.T.shape) - self.S + self.F2)

        # Combine the result for final objective function
        ans += self.parameter['beta1'] * norm(self.V.flatten(), 1) + self.parameter['beta2'] * norm(self.S.flatten(), 1) +  augment_part1 + augment_part2 
        return ans
    
    def fit(self, gamma1, gamma2, gamma3, eta = 0.0001, maxitor = 50):
        """
        Fiiting anomaly detection models
        (This function estimates the low rank and sparse part in tensor)

        In general, this function also works as an estimator for robust tensor PCA

        optional input:
        eta: average difference (between each iteration, converge)
        maxitor: maximum number of iteration

        gamma1, gamma2, and gamma3 are learning rates for two augmented lagrangian multipliers

        This function also records the objective function value in fly
        """
        print("Start fitting robust tensor PCA model...\n")
        self.Objective_value = [self.get_Objective()]

        threshold_lambda = 2 * self.parameter['lambda1'] / self.parameter['rho3']
        threshold_V = 2 * self.parameter['beta1'] / (3 * self.parameter['rho3'] + self.parameter['rho1'])
        threshold_S = 2 * self.parameter['beta2'] / self.parameter['rho2']

        
        current_eta, current_itor = 1000, 0
        
        bar = progressbar.ProgressBar(maxval=maxitor, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        while current_eta > eta and current_itor < maxitor:
            
            prev_obj = self.get_Objective()
            
            self.Wlist = soft_HOSVD(self.X, self.V, threshold_lambda, self.parameter['phi'], self.F3)
            # Update W
            self.W = sum(self.Wlist) / len(self.Wlist)

            # Update V
            tensor_Tau = self.parameter['rho3'] * sum([self.X - self.Wlist[i] + self.F3[i] for i in range(len(self.F3))]) + self.parameter['rho1'] * (self.T - self.F1)
            tensor_Tau /= (3 * self.parameter['rho3'] + self.parameter['rho1'])
            self.V = soft_threshold(tensor_Tau, threshold_V)

            # Update S
            tensor_T = tl.fold(np.dot(self.Dmatrix, tl.unfold(self.T, 0)), 0, self.X.shape) + self.F2
            self.S = soft_threshold(tensor_T, threshold_S)

            # Update T using gradient descent
            tmp_inv = np.linalg.pinv(self.parameter['rho1'] * np.eye(self.T.shape[0]) + self.parameter['rho2'] * np.dot(self.Dmatrix.T, self.Dmatrix))
            tmp_right = self.parameter['rho1'] * tl.unfold(self.V + self.F1, 0) + self.parameter['rho2'] * np.dot(self.Dmatrix.T, tl.unfold(self.S - self.F2, 0))
            self.T = tl.fold(np.dot(tmp_inv, tmp_right), 0, self.T.shape)


            # Update Dual parameters
            self.F1 += gamma1 * (self.V - self.T)
            self.F2 += gamma2 * (tl.fold(np.dot(self.Dmatrix, tl.unfold(self.T, 0)), 0, self.T.shape) - self.S)
            for i in range(len(self.F3)):
                self.F3[i] += gamma3 * (self.X - self.Wlist[i] - self.V)

            
            current_obj = self.get_Objective()
            self.Objective_value.append(current_obj)
            current_eta = np.log(prev_obj - current_obj)

            bar.update(current_itor + 1)
            current_itor += 1
        bar.finish()
        return

    def Find_Anomalies(self, threshold = 700):
        """Find Anomalies by calculating mahalanobis distance
        
        Robust covariance matrix is estimated by Minimum determinant Covaraince (robust estimate)
        Mahalanobis distance is calculated with the robust covariance matrix
        First threshold points are selected as anomalies with largest mahalanobis distance

        Note: This function only works with three modes tensor

        Input:
        threshold: how many potential anomalies are considered; default is 700
        """

        if self.anomaliscores is None:
            # if anomaly scores are not available, compute anomaly scores first
            self.anomaliscores = np.zeros(self.X.shape)
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[-1]):
                    tmp_data = np.squeeze(self.V[i, :, j]).reshape(-1, 1)
                    try:
                        tmp_Cov = MinCovDet().fit(tmp_data)                     #reshape the squeezed vector so that it contains only one features
                        # Calculate sqrt root of mahalanobis and save as anomalies socres
                        self.anomaliscores[i, :, j] = tmp_Cov.mahalanobis(tmp_data - tmp_Cov.location_) **(0.33)
                    except ValueError:
                        # If the fiber is all zero, 
                        self.anomaliscores[i, :, j] = np.zeros(self.X.shape[1])
        anomalies = np.unravel_index(np.argsort(-1 * self.anomaliscores, axis = None)[: threshold], self.anomaliscores.shape)
        return anomalies

    def plot_history(self):
        """Plot the value of objective function as model is fitted
        
        The model has to be fitted first
        """
        plt.figure(figsize=(12, 9))
        plt.plot(self.Objective_value)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title('Objective Function Values')
        # plt.savefig(filename)
        plt.show()
        return 

"""The result from Find_Anomalies can be compared with true value.

We create a new function to compare anomalies and plot accuracy figures"""

def anomaly_acc(pred_anomaly, true_anomaly):
    """ 
    Compare predicted anomalies and true anomalies by computing the accuracy 

    both predicted anomaly positions and true anomaly positions are indices for three mode tensor
    predicted anomalies are in a hash set
    """
    total_pos = sum(true_anomaly.flatten())
    true_pos = sum(true_anomaly[pred_anomaly])
    recall = true_pos / total_pos
    pred_pos = len(np.ravel_multi_index(pred_anomaly, true_anomaly.shape))
    precision = true_pos / pred_pos
    return precision, recall

def anomaly_ROC(pred_anomaly, true_anomaly):
    """
    Calculate TPR and FPR to create ROC curve
    """
    total_pos = sum(true_anomaly.flatten())
    tpr = sum(true_anomaly[pred_anomaly]) / total_pos
    false_pos = len(np.ravel_multi_index(pred_anomaly, true_anomaly.shape)) - sum(true_anomaly[pred_anomaly])
    fpr = false_pos / (np.prod(true_anomaly.shape) - total_pos)
    return tpr, fpr