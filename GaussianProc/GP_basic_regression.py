import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Set the seed
np.random.seed(1)

# Define the function to predict
def f(x):
    return x*np.sin(x)*(1+np.cos(x))

#-------------------------------------------------------------------------------
# Without noise
X = np.atleast_2d([1.,3.,5.,6.,7.,8.,10,12,14,15]).T
# Observations
y = f(X).ravel()
# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0,15,1000)).T

# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)


# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X,y)

# Predict on the meshed x-axis and ask for MSE
y_pred, sigma = gp.predict(x, return_std = True)

# Plot the function, the prediction and the 95% CI based on the MSE
plt.figure(1)
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)(1+\cos(x))$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')

plt.fill(np.concatenate([x, x[::-1]]), np.concatenate([y_pred - 1.9600 * sigma,(y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-20, 20)
plt.legend(loc='upper left')
plt.savefig('Noiseless.pdf')

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# With noise

X1 = np.atleast_2d(np.linspace(0.1,15,30)).T

# Observations and noise
y1 = f(X1).ravel()
dy1 = 0.5+1.0*np.random.random(y1.shape)
noise = np.random.normal(0,dy1)
y_new = y1 + noise

# Instantiate a Gaussian process model
GP = GaussianProcessRegressor(kernel=kernel, alpha = dy1**2, n_restarts_optimizer = 10)

# Fit to data using Maximum Likelihood estimation
GP.fit(X1,y_new)

# Make the prediction
y_pred_new, sigma_new = GP.predict(x, return_std = True)

plt.figure(2)
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)(1+\cos(x))$')
plt.errorbar(X1.ravel(), y_new, dy1, fmt='r.', markersize=10, label='Observations')
plt.plot(x,y_pred_new,'b-',label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred_new - 1.9600 * sigma_new,
                        (y_pred_new + 1.9600 * sigma_new)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-20, 20)
plt.legend(loc='upper left')
plt.savefig('Noise.pdf')

plt.show()













#-------------------------------------------------------------------------------