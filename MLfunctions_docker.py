import numpy as np
import pandas as pd
import os 
from sklearn.linear_model import LogisticRegression, LinearRegression, QuantileRegressor

from sklearn.utils.fixes import parse_version, sp_version
from sklearn.ensemble import RandomForestRegressor 

from conquer_linear import low_dim


import gurobipy as gp
from gurobipy import Model, GRB


options = {
    "WLSACCESSID": '',
    "WLSSECRET": '',
    "LICENSEID": ,
}





###################################################



def propensity_score_logistic(X, A,random_state=42):
    model = LogisticRegression(random_state=random_state)
    model.fit(X, A)
    return model

# use model.predict_proba(X_test) 
# will have two columns if you have a binary classification problem (assuming you have two classes). 
# The first column corresponds to the probability of the sample belonging to the negative class (class 0), 
# and the second column corresponds to the probability of the sample belonging to the positive class (class 1).



###################################################




def mu1_linear(X, A, Y):
    Xmu1 = X[A == 1]
    Ymu1 = Y[A == 1]
    # print(Xmu1.shape)
    model = LinearRegression()
    model.fit(Xmu1, Ymu1)
    return model

def mu0_linear(X, A, Y):
    Xmu0 = X[A == 0]
    Ymu0 = Y[A == 0]
    # print(Xmu0.shape)
    model = LinearRegression()
    model.fit(Xmu0, Ymu0)
    return model


def mu1_wls_linear(X, A, Y):
    Xmu1 = X[A == 1]
    Ymu1 = Y[A == 1]
    # print(Xmu1.shape)

    # Step 1: Fit a preliminary linear model to estimate residuals
    preliminary_model = LinearRegression()
    preliminary_model.fit(Xmu1, Ymu1)
    preliminary_predictions = preliminary_model.predict(Xmu1)
    residuals = Ymu1 - preliminary_predictions
    
    # Step 2: Estimate the standard deviation of residuals as a linear function of X
    abs_residuals_model = LinearRegression()
    abs_residuals_model.fit(Xmu1, np.abs(residuals))
    estimated_std_dev = abs_residuals_model.predict(Xmu1)
    
    # Prevent division by zero or extremely small std dev estimates
    estimated_std_dev = np.clip(estimated_std_dev, a_min=1e-5, a_max=None)
    
    # Step 3: Compute weights as the inverse of the variance (std dev squared)
    weights = 1.0 / np.square(estimated_std_dev)
    
    # Step 4: Fit the weighted least squares model
    wls_model = LinearRegression()
    wls_model.fit(Xmu1, Ymu1, sample_weight=weights)
    return wls_model

def mu0_wls_linear(X, A, Y):
    Xmu0 = X[A == 0]
    Ymu0 = Y[A == 0]
    # print(Xmu1.shape)

    # Step 1: Fit a preliminary linear model to estimate residuals
    preliminary_model = LinearRegression()
    preliminary_model.fit(Xmu0, Ymu0)
    preliminary_predictions = preliminary_model.predict(Xmu0)
    residuals = Ymu0 - preliminary_predictions
    
    # Step 2: Estimate the standard deviation of residuals as a linear function of X
    abs_residuals_model = LinearRegression()
    abs_residuals_model.fit(Xmu0, np.abs(residuals))
    estimated_std_dev = abs_residuals_model.predict(Xmu0)
    
    # Prevent division by zero or extremely small std dev estimates
    estimated_std_dev = np.clip(estimated_std_dev, a_min=1e-5, a_max=None)
    
    # Step 3: Compute weights as the inverse of the variance (std dev squared)
    weights = 1.0 / np.square(estimated_std_dev)
    
    # Step 4: Fit the weighted least squares model
    wls_model = LinearRegression()
    wls_model.fit(Xmu0, Ymu0, sample_weight=weights)
    return wls_model

###################################################

def Gamma_1(X, A, Y, mu1_model, logis_model):
    mu1_values = mu1_model.predict(X)
    prob1_values = logis_model.predict_proba(X)[:,1]
    Gamma_1_values = mu1_values + (Y - mu1_values) * (A == 1)/prob1_values

    return Gamma_1_values 



def Gamma_0(X, A, Y, mu0_model, logis_model):
    mu0_values = mu0_model.predict(X)
    prob0_values = logis_model.predict_proba(X)[:,0]
    Gamma_0_values = mu0_values + (Y - mu0_values) * (A == 0)/prob0_values

    return Gamma_0_values 



###################################################




def qtau_linear(tau, X, A, Y):

    # This line is to avoid incompatibility if older SciPy version.
    # You should use `solver="highs"` with recent version of SciPy.
    solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

    qr = QuantileRegressor(quantile=tau, alpha=0, solver=solver)

    AX = pd.concat([A,X], axis = 1)
    qr.fit(AX, Y)
    return qr 
# When predicting, use .predict(AX)


# def psi_d(tau, X, A, Y, qr_pred, pi)

###################################################

def exp_kernel_generator(h=1):
    return lambda x: 1/h * np.exp(-x**2/h**2/2)

cond_density_kernel = exp_kernel_generator(h=1)


def cond_density_model(qtau_model, X, A, Y):

    AX = pd.concat([A,X], axis = 1)

    qtaua_out = pd.Series(qtau_model.predict(AX))
    model0 = RandomForestRegressor(n_estimators=50, 
                                  max_depth=5, 
                                  min_samples_leaf=5, 
                                  min_samples_split=10)
    model1 = RandomForestRegressor(n_estimators=50, 
                                max_depth=5, 
                                min_samples_leaf=5, 
                                min_samples_split=10)
    nested_outcome_func = lambda nuis, Y: cond_density_kernel(Y - nuis)
    
    nested_outcome = nested_outcome_func(qtaua_out, Y)

    model0.fit(X[A==0], nested_outcome[A==0])
    model1.fit(X[A==1], nested_outcome[A==1])

    return model0, model1 





################################################### 
#  optimization 




def optimi_solve_0217(x_values, Gamma_1, Gamma_0,Psi_1,Psi_0, q, ttlimit = GRB.INFINITY):
    #

    #  dim include the intercept, x_values do not. So if x_value is X_1, X_2, then we need dim = 3 to include the intercept.

    #
    X = np.array(x_values)
    a = np.array(Gamma_1) - np.array(Gamma_0)
    b = np.array(Psi_1) - np.array(Psi_0)
    c = np.sum(np.array(Psi_0))

    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((ones_column, X))

    dim = X.shape[1] #include the intercept 
    n = len(X)


    with gp.Env(params=options) as env, gp.Model(env=env) as model:
    # Create a model
        # model = Model("Optimization")

        model.Params.timeLimit = ttlimit 

        # Define variables
        beta = model.addVars(dim, lb=-GRB.INFINITY, name="beta")
        z = model.addVars(n, vtype=GRB.BINARY, name="z")

        # Big-M formulation constants (need to be large enough)
        M = 1e6

        # Objective function
        model.setObjective(sum(a[i] * z[i] for i in range(n)), GRB.MAXIMIZE)

        # Constraints
        for i in range(n):
            model.addConstr(sum(X[i, j] * beta[j] for j in range(dim)) <= M * z[i])
            model.addConstr(sum(X[i, j] * beta[j] for j in range(dim)) >= -M * (1 - z[i]))

        # Mean constraint
        model.addConstr(sum(b[i] * z[i] for i in range(n))  >= (n*q - c))

        model.addConstr(sum(beta[i] * beta[i] for i in range(dim)) == 1, "L2_norm_constraint")


        # Solve the model
        model.optimize()

        # Retrieve the solution
        beta_optimal = [beta[j].X for j in range(dim)]

    return beta_optimal



# this can be used to find the one maximizes the mean or the one maximizes the average of quantile

def optimi_solve_nc_0206(x_values, Gamma_1, Gamma_0, ttlimit = GRB.INFINITY):
    X = np.array(x_values)
    a = np.array(Gamma_1) - np.array(Gamma_0)

    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((ones_column, X))

    dim = X.shape[1]  #include the intercept 
    n = len(X)
    # Create a model
    with gp.Env(params=options) as env, gp.Model(env=env) as model:
    # model = Model("Optimization")

        model.Params.timeLimit = ttlimit 

        # Define variables
        beta = model.addVars(dim, lb=-GRB.INFINITY, name="beta")
        z = model.addVars(n, vtype=GRB.BINARY, name="z")

        # Big-M formulation constants (need to be large enough)
        M = 1e6

        # Objective function
        model.setObjective(sum(a[i] * z[i] for i in range(n)), GRB.MAXIMIZE)

        # Constraints
        for i in range(n):
            model.addConstr(sum(X[i, j] * beta[j] for j in range(dim)) <= M * z[i])
            model.addConstr(sum(X[i, j] * beta[j] for j in range(dim)) >= -M * (1 - z[i]))


        model.addConstr(sum(beta[i] * beta[i] for i in range(dim)) == 1, "L2_norm_constraint")

        # Solve the model
        model.optimize()

        # Retrieve the solution
        beta_optimal = [beta[j].X for j in range(dim)]

    return beta_optimal


