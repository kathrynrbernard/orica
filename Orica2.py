
# coding: utf-8

# In[528]:


### imports

import mne
import numpy as np
import pandas as pd
import scipy
import scipy.io
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[529]:


class Orica():
    def __init__(self,
                 weights=None,
                 num_chans=32,
                 sphering=False,
                 numpass=1,
                 block_ica=8,
                 block_white=8,
                 forgetfac="cooling",
                 localstat=np.inf,
                 ffdecayrate=0.6,
                 nsub=0,
                 evalconverg=0,
                 verbose=True,
                 nlfunc=[]):

        self.num_pass = numpass
        if weights is None:
            self.state_icaweights = np.eye(num_chans)
        else:
            self.state_weights = weights
        self.verbose = verbose

        self.block_ica = block_ica
        self.num_subgaussian = nsub

        self.block_white = block_white
        self.online_whitening = sphering

        self.FF_profile = forgetfac
        self.FF_tauconst = localstat
        self.FF_gamma = ffdecayrate
        self.FF_lambda_0 = 0.995
        self.FF_decay_rate_alpha = 0.02
        self.FF_upper_bound_beta = 0.001
        self.FF_trans_band_width_gamma = 1
        self.FF_trans_band_center = 5
        self.FF_lambda_init = 0.1

        self.eval_converge_profile = evalconverg
        self.eval_converge_leaky_avg_delta = 0.01
        self.eval_converge_leaky_avg_delta_var = 0.001

        if self.online_whitening:
            self.state_icasphere= np.eye(num_chans)
            self.state_icasphere = np.matrix(self.state_icasphere)

        self.state_lambda_k = np.zeros((1, self.block_ica), dtype=float)
        self.state_min_non_stat_idx = []
        self.state_counter = 0

        if (self.FF_profile == "cooling") or (self.FF_profile == "constant"):
            self.FF_lambda_const = 1 - np.exp(-1 / self.FF_tauconst)

        if self.eval_converge_profile:
            self.state_rn = []
            self.state_nonstatidx = []

        self.state_kurtsign = np.ones((num_chans, 1), dtype=float) > 0
        if self.num_subgaussian != 0:
            self.state_kurtsign[0:self.num_subgaussian] = False

        self.nlfunc = nlfunc


# In[530]:


def ica(self, full_data):
        # not sure if this next line is necessary for all datasets
        # might just convert to matrix without dropping a column
        data = np.matrix(full_data.iloc[0:-1,:])  # dropping the last row (STI 014) - all zeros, not in EEGLAB EEG.data
        #print('python_data_init', data)
        num_chans, num_points = data.shape
        if not self.online_whitening:
            if self.verbose:
                print("Use pre-whitening method\n")
            self.state_icasphere = 2.0 * np.linalg.inv(scipy.linalg.sqrtm(np.cov(data)))
            print('sphere:', 2.0 * np.linalg.inv(scipy.linalg.sqrtm(np.cov(data))))
            self.state_icasphere = np.matrix(self.state_icasphere)
        else:
            if self.verbose:
                print("Use online whitening method\n")
        data = self.state_icasphere * data
        num_blocks = np.floor(num_points / np.min([self.block_ica, self.block_white]))
        if self.verbose:
            print_flag = 0
        for i in range(self.num_pass):
            for bi in range(int(num_blocks)):
                data_range = np.arange((np.floor(bi*num_points/num_blocks)),
                                        np.min([num_points,
                                                np.floor((bi + 1) * num_points/num_blocks)]),
                                        dtype=int) # use so you don't get index out of range
                data_range1 = np.arange((np.floor(bi*num_points/num_blocks)),
                                        1 + np.min([num_points,
                                                    np.floor((bi + 1) * num_points/num_blocks)]),
                                        dtype=int)
                data_range1 = data_range1[1:]  # use so that you don't have 0s in equations (gives infs)
                # matlab dataRange is 1 2 3 4 5 6 7 8
                # python indexing starts at 0 so it's 0 1 2 3 4 5 6 7
                if self.online_whitening:
                    self.state = self.dynamicWhitening(data[:, data_range], data_range1)
                    data[:, data_range] =  self.state_icasphere * data[:, data_range]
                self.state = dynamicOrica(self, data[:, data_range], data_range1)
                if self.verbose:
                    if print_flag < (np.floor(10
                                            * ((i - 1) * num_blocks + bi)
                                            / self.num_pass
                                            / num_blocks)):
                        print_flag = print_flag + 1
                        print(" %d%% ", 10 * print_flag)
        weights = self.state_icaweights
        sphere = self.state_icasphere
        
        data_init_mat = scipy.io.loadmat('dataInit.mat', squeeze_me=True)
        data_init_a = data_init_mat['data']
        
        data_final_mat = scipy.io.loadmat('dataFinal.mat', squeeze_me=True)
        data_final_a = data_final_mat['data2']
        
        data_final_dif = np.subtract(data, data_final_a)
        
        return(sphere)


# In[531]:


def dynamicWhitening(self, block_data, data_range):
        num_points = block_data.shape[1]
        if self.FF_profile == "cooling":
            lam = self.genCoolingFF(self.state_counter+data_range)
            if lam[0] < self.FF_lambda_const:
                lam = np.matlib.repmat(self.FF_lambda_const, 1, num_points)
        elif self.FF_profile == "constant":
            lam = np.matlib.repmat(self.FF_lambda_const, 1, num_points)
        elif self.FF_profile == "adaptive":
            lam = np.matlib.repmat(self.state_lambda_k[-1], 1, num_points)
        v = self.state_icasphere * block_data
        lambda_avg = 1 - lam[np.int(np.ceil(lam[-1] / 2))]
        q_white = lambda_avg / (1 - lambda_avg) + np.trace((v.getH() * v) / num_points)
        self.state_icasphere = (1 / lambda_avg
                                * (self.state_icasphere
                                    - v
                                    * np.transpose(v)
                                    / num_points
                                    / q_white
                                    * self.state_icasphere))
        
        return(self.state_icasphere)


# In[532]:


def dynamicOrica(self, block_data, data_range):
        num_chans, num_points = block_data.shape
        f = np.zeros((num_chans, num_points), dtype=float)
        self.state_icaweights = np.matrix(self.state_icaweights)
        y = self.state_icaweights * block_data
        self.state_kurtsign = np.multiply(self.state_kurtsign, 1)  # converting from true/false to 1/0
        if not self.nlfunc:
            f[:] = -2 * np.tanh(np.asarray(y[:]))
            f[self.state_kurtsign,:] = -2 * np.tanh(np.asarray(y[self.state_kurtsign,:]))
        else:
            f = self.nlfunc(y)
        if self.eval_converge_profile:
            model_fitness = np.identity(num_chans) + y * np.transpose(f) / num_points
            var = np.dot(block_data, block_data)
            if not self.state_rn:
                self.state_rn = model_fitness
            else:
                self.state_rn = ((1 - self.eval_converge_leaky_avg_delta)
                                    * self.state_rn
                                    + self.eval_converge_leaky_avg_delta
                                    * model_fitness)
            self.state_nonstatidx = np.linalg.norm(self.state_rn, "fro")
        if self.FF_profile == "cooling":
            self.state_lambda_k = genCoolingFF(self, self.state_counter + data_range)
            if self.state_lambda_k[1] < self.FF_lambda_const:
                self.state_lambda_k = np.matlib.repmat(self.FF_lambda_const, 1, num_points)
            self.state_counter = self.state_counter + num_points
        elif self.FF_profile == "constant":
            self.state_lambda_k = np.matlib.repmat(self.FF_lambda_const, 1, num_points)
        elif self.FF_profile == "adaptive":
            if not self.state_min_non_stat_idx:
                self.state_min_non_stat_idx = self.state_nonstatidx
            self.state_min_non_stat_idx = (max(min(self.state_min_non_stat_idx,
                                                    self.state_nonstatidx), 1))
            ratio_norm_rn = self.state_nonstatidx / self.state_min_non_stat_idx
            self.state_lambda_k = self.genAdaptiveFF(data_range,
                                                    ratio_norm_rn)

        lambda_prod = np.prod(np.divide(1, (1 - self.state_lambda_k)))
        q = 1 + (self.state_lambda_k * np.diagonal(np.inner(f.T, y.T) - 1))
        diag = np.diag(np.divide(self.state_lambda_k, q))
        diag = y * diag
        self.state_icaweights = (np.multiply(lambda_prod,
                                            (self.state_icaweights
                                            - np.matmul(diag, np.transpose(f))
                                            * self.state_icaweights)))

        D_val, V_val = (np.linalg.eig(self.state_icaweights
                                      * np.transpose(self.state_icaweights)))
        #  manually re-ordered eigenvalues based on matlab output
        # (moved first two to the end)
        D_val_first = D_val[0]
        D_val_second = D_val[1]
        D_val = np.append(D_val[2:], (D_val_second, D_val_first))
        D = np.zeros((num_chans, num_chans), dtype=float)
        np.fill_diagonal(D, D_val) # modified in-place
        # re-ordered eigenvectors in the same way
        V_val_first = V_val[:,0]
        V_val_second = V_val[:,1]
        V = V_val[:,2:]
        V = np.hstack((V, V_val_second, V_val_first))
        self.state_icaweights = (V
                                * np.linalg.inv(np.sqrt(D))
                                * np.transpose(V)
                                * self.state_icaweights)
        
        return(self.state_icaweights)


# In[533]:


def genCoolingFF(self, t):
        lam = np.divide(self.FF_lambda_0, np.power(t, self.FF_gamma))
        
        return(lam)


# In[534]:


def genAdaptiveFF(self, data_range, ratio_norm_rn):
        gain_for_errors = (self.FF_upper_bound_beta
                            * 0.5
                            * (1 + tanh((ratio_norm_rn - self.FF_trans_band_center)
                                / self.FF_trans_band_width_gamma)))
        f = lambda n : (np.power((1 + gain_for_errors), n)
                                    * self.state_lambda_k[-1]
                                    - self.FF_decay_rate_alpha
                                    * (np.power((1 + gain_for_errors), (2 * n - 1))
                                        - np.power((1 + gain_for_errors), (n - 1)))
                                    / gain_for_errors
                                    * (self.state_lambda_k[-1] ** 2))
        self.state_lambda_k = f[1:len(data_range)]
        
        return(self.state_lambda_k)


# In[535]:


if __name__ == "__main__":
    raw_data = mne.io.read_raw_eeglab("/Users/kathrynbernard/Desktop/summer 2018/sederberg lab/orcia/SIM_STAT_16ch_3min.set")
    raw_data.load_data()
    data_df = raw_data.to_data_frame()
    data_df = data_df.transpose()
    data_init = Orica(num_chans=16)  # initialize class obj using most default values, specify number of channels
    analyzed_data = ica(data_init, full_data=data_df)
    
    weights_mat = scipy.io.loadmat('weights.mat', squeeze_me=True)
    weights_a = weights_mat['weights']
    
    weights_dif = np.subtract(analyzed_data, weights_a)
   
    plt.plot(analyzed_data, data=analyzed_data)
    
    
    
    
   
    

