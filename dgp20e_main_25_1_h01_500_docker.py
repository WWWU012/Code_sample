import numpy as np
import pandas as pd
import os 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold
from scipy.stats import norm
import time 



import sys 
sys.path.append('/app')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#import statsmodels.api as sm
from conquer_linear import low_dim

from MLfunctions_docker import *
from MLfunctions_2 import *

#timelimit for each optimization 
ttlimit = 1200

def main(i):

    print(f"Processing iteration {i}")

    # Start the timer for this iteration
    # start_time_iteration = time.time()
    df = pd.read_csv(f'data_{i}_0.25.csv')
    dimX = 2

    A = df['A']
    X = df.iloc[:,0:dimX]
    Y = df['Y']

    Y_tau_0 = df['Y_tau_0']
    Y_tau_1 = df['Y_tau_1']
    Y_mean_0 = df['Y_mean_0']
    Y_mean_1 = df['Y_mean_1']
    A_prob = df['A_prob']


    probab_ora = np.where(A ==1, A_prob, 1 - A_prob)
    AIPW_1 = Y_mean_1 + (Y - Y_mean_1)*(A == 1)/probab_ora
    AIPW_0 = Y_mean_0 + (Y - Y_mean_0)*(A == 0)/probab_ora   

#### cross-fitting 
    k = 5 
    kf =KFold(n_splits=k)


    tau = 0.25    
    h = 0.1
    n = len(A)
    Gamma_1 = np.zeros(n)
    Gamma_0 = np.zeros(n)
    mu1_pre = np.zeros(n)
    mu0_pre = np.zeros(n)
    q_1 = np.zeros(n)
    q_0 = np.zeros(n)

    psi_hat_1 = np.zeros(n)
    psi_hat_0 = np.zeros(n)




    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        A_train, A_test = A[train_index], A[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model_p_train = propensity_score_logistic(X_train, A_train)
        probab_test = model_p_train.predict_proba(X_test)
    # probab_a = np.where(A ==1, probab[:,1], probab[:,0])

        mu1_rf_train = mu1_rf(X_train, A_train, Y_train)
        mu0_rf_train = mu0_rf(X_train, A_train, Y_train)

        mu1_pre_test = mu1_rf_train.predict(X_test)
        mu0_pre_test = mu0_rf_train.predict(X_test)

        Gamma_1[test_index] = mu1_pre_test + (Y_test - mu1_pre_test)*(A_test==1)/probab_test[:,1]
        Gamma_0[test_index] = mu0_pre_test + (Y_test - mu0_pre_test)*(A_test==0)/probab_test[:,0]

        mu1_pre[test_index] = mu1_pre_test
        mu0_pre[test_index] = mu0_pre_test

     # qr_1 = qr_fit_1(X, A, Y, tau) # model
    # qr_0 = qr_fit_0(X, A, Y, tau) # model

        qr_1_train = qr_fit_1_rf(X_train, A_train, Y_train) # model
        qr_0_train = qr_fit_0_rf(X_train, A_train, Y_train) # model  

    # q_star = norm.ppf(tau)

    # q_1 = qr_pred(qr_1, X) # q_1 values 
    # q_0 = qr_pred(qr_0, X) # q_0 values

        q_1_test = qr_pred_rf(qr_1_train, X_test, tau) # q_1 values 
        q_0_test = qr_pred_rf(qr_0_train, X_test, tau) # q_0 values

        q_1[test_index] = q_1_test
        q_0[test_index] = q_0_test

    # f1 = cond_density_1(X, A, Y, tau, h)
    # f0 = cond_density_0(X, A, Y, tau, h)

        f1_train = cond_density_1_rf_pre(X_train, A_train, Y_train, tau, h, X_test)
        f0_train = cond_density_0_rf_pre(X_train, A_train, Y_train, tau, h, X_test)

    # psi_hat_1 = psi_1(X, Y, A, tau, probab[:,1], q_1, f1)
    # psi_hat_0 = psi_0(X, Y, A, tau, probab[:,0], q_0, f0)
        psi_hat_1_test = psi_1(X_test, Y_test, A_test, tau, probab_test[:,1], q_1_test, f1_train)
        psi_hat_0_test = psi_0(X_test, Y_test, A_test, tau, probab_test[:,0], q_0_test, f0_train)

        psi_hat_1[test_index] = psi_hat_1_test
        psi_hat_0[test_index] = psi_hat_0_test

    
    #q_values = [0, 2.35]
    
    q_values = [-2] 
    
    # Define the new folder name where you want to save the CSV files
    new_folder_name = './dgp20_result25_h01_ad_500'

    # Check if the new folder exists, if not, create it
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)

    for q in q_values:
        try:
            betahat_nc_ora = optimi_solve_nc_0206(X, Y_mean_1, Y_mean_0, ttlimit)
            betahat_nc_ora_df = pd.DataFrame(betahat_nc_ora)
            betahat_nc_ora_df.to_csv(f'{new_folder_name}/betahats_result_nc_ora_25_{q}_iter_{i}.csv', index=False)
        except Exception as e:
            print(f"An error occurred: {e}")


        try:
            betahat_nc_aipw = optimi_solve_nc_0206(X, Gamma_1, Gamma_0, ttlimit)
            betahat_nc_aipw_df = pd.DataFrame(betahat_nc_aipw)
            betahat_nc_aipw_df.to_csv(f'{new_folder_name}/betahats_result_nc_aipw_25_{q}_iter_{i}.csv', index=False)
        except Exception as e:
            print(f"An error occurred: {e}")
 


        try:
            betahat_nc_lin = optimi_solve_nc_0206(X, mu1_pre, mu0_pre, ttlimit)
            betahat_nc_lin_df = pd.DataFrame(betahat_nc_lin)
            betahat_nc_lin_df.to_csv(f'{new_folder_name}/betahats_result_nc_lin_25_{q}_iter_{i}.csv', index=False)
        except Exception as e:
            print(f"An error occurred: {e}")



        try:
            betahat_qt_dr = optimi_solve_nc_0206(X, psi_hat_1, psi_hat_0, ttlimit)
            betahat_qt_dr_df = pd.DataFrame(betahat_qt_dr)
            betahat_qt_dr_df.to_csv(f'{new_folder_name}/betahats_result_qt_dr_25_{q}_iter_{i}.csv', index=False)
        except Exception as e:
            print(f"An error occurred: {e}")

        try:
            betahat_qt = optimi_solve_nc_0206(X, q_1, q_0, ttlimit)
            betahat_qt_df = pd.DataFrame(betahat_qt)
            betahat_qt_df.to_csv(f'{new_folder_name}/betahats_result_qt_25_{q}_iter_{i}.csv', index=False)
        except Exception as e:
            print(f"An error occurred: {e}")

        try:
            betahat_qt_ora = optimi_solve_nc_0206(X, Y_tau_1, Y_tau_0, ttlimit)
            betahat_qt_ora_df = pd.DataFrame(betahat_qt_ora)
            betahat_qt_ora_df.to_csv(f'{new_folder_name}/betahats_result_qt_ora_25_{q}_iter_{i}.csv', index=False)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Check if an argument is passed
    if len(sys.argv) > 1:
        iteration = int(sys.argv[1])  # Get the iteration number from command line argument
        main(iteration)
    else:
        print("No iteration number provided.")
