# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pystan
import GPy.kern as gp_kern
import GPy as GPy
import matplotlib.pyplot as plt
from sklearn.gaussian_process import kernels as sk_kern
from sklearn.gaussian_process import GaussianProcessRegressor as GPR


def plot_result(x_test, mean, std):
    plt.plot(x_test[:,0], mean, color="C0", label="predict mean")
    plt.fill_between(x_test[:,0], mean + std, mean - std, color="C0", alpha=.3,label= "1 sigma confidence")




#sklearnによるGaussianProcess
x_train = np.random.normal(0, 1.0, 20).reshape(-1,1)
y_train = x_train + np.sin(4 * x_train) + np.random.normal(loc=0, scale=0.1, size=x_train.shape)
x_test=np.linspace(-3., 3., 200).reshape(-1, 1)


kernel = sk_kern.RBF(1.0, (1e-3, 1e3)) + sk_kern.ConstantKernel(1.0, (1e-3, 1e3)) + sk_kern.WhiteKernel()

clf = GPR(
    kernel=kernel,
    alpha=1e-10, 
    optimizer="fmin_l_bfgs_b", 
    n_restarts_optimizer=20,
    normalize_y=True)
    
clf.fit(x_train.reshape(-1, 1), y_train)
pred_mean, pred_var=clf.predict(x_test, return_std=True)
plot_result(x_test=x_test,mean=pred_mean[:,0],std=pred_var)
plt.title("Scikit-learn")
plt.legend()
plt.savefig("sklern_predict.png", dpi=150)
plt.close("all")


#Gpy
kern = gp_kern.RBF(input_dim=1) + gp_kern.Bias(input_dim=1)+gp_kern.PeriodicExponential(input_dim=1)
gpy_model = GPy.models.GPRegression(X=x_train.reshape(-1, 1), Y=y_train, kernel=kern, normalizer=None)
gpy_model.optimize()
pred_mean, pred_var = gpy_model.predict(x_test.reshape(-1, 1), )
pred_std = pred_var ** 0.5
plot_result(x_test, mean=pred_mean[:, 0], std=pred_std[:, 0])
plt.legend()
plt.title("GPy")
plt.savefig("GPy_predict.png", dpi=150)
plt.close("all")