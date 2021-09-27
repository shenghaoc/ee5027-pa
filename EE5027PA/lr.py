# Q3. Logistic regression
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from numpy import linalg as LA

# 57 columns for X, 1 column for y
# 3065 for training, 1536 for test
# Need to squeeze out y from single-vaue ndarray
spam_contents = sio.loadmat('spamData.mat', squeeze_me=True)

# Do preprocessing and use unqualified name for brevity
# Log-transform
Xtrain = np.log(np.add(spam_contents['Xtrain'], 0.1))
Xtest = np.log(np.add(spam_contents['Xtest'], 0.1))
ytrain = spam_contents['ytrain']
ytest = spam_contents['ytest']

# Add bias term 1 to  start of X
# N * D matrix to N * (D+1) matrix
# 57 to 58
Xtrain_biased = np.insert(Xtrain, 0, 1, axis=1)
Xtest_biased = np.insert(Xtest, 0, 1, axis=1)

h_mask = np.identity(Xtrain_biased.shape[1])
h_mask[0, 0] = 0
g_mask = np.ones(Xtrain_biased.shape[1])
g_mask[0] = 0


def sigm(x):
    return 1 / (1 + np.exp(-x))


# Just try to match the dimensions, row major order in code makes things unintuitive
# last dimension of x1 should match second-to-last dimension of x2
def get_gradient(X, mu):
    # X^T (mu - y) (58,3065) * (58,1)
    g = np.matmul(X.transpose(), mu - ytrain)
    return g


def get_hessian(X, mu):
    # S is zeros except for diagonals
    s = np.diag(mu * (1 - mu))
    # X^T S, (58, 3065) * (3065, 3065)
    h = np.matmul(X.transpose(), s)
    # (X^T S)X, (58, 3065) * (3065, 58)
    h = np.matmul(h, X)
    return h


def calc_err(lam, X, y, mode):
    max_iter = 10000
    tolerance = 1e-7
    # Init (D+1) zero vector to match Xtrain_biased
    # Should be 58
    w = np.zeros(Xtrain_biased.shape[1])
    for n in range(0, max_iter):
        # w^T X, (58,1) * (58,3065)
        mu = sigm(np.matmul(w, Xtrain_biased.transpose()))
        g = get_gradient(Xtrain_biased, mu)
        h = get_hessian(Xtrain_biased, mu)

        # l_2 regularization
        g_reg = g + lam * w * g_mask
        h_reg = h + lam * h_mask

        # w_{k+1} = w_k - H^{-1} g_k
        diff = -np.matmul(LA.inv(h_reg), g_reg)
        if np.abs(np.mean(diff)) < tolerance:
            break
        w += diff

    # Training Set
    p_y_1 = sigm(np.matmul(X, w))
    err = 1 - np.sum((p_y_1 >= 0.5) == y) / y.size
    if lam == 1 or lam == 10 or lam == 100:
        print('At lambda =', lam, mode, 'error rate =', err)
    return err


def plot():
    print('Running a logistic regression model')
    lam_vals = np.r_[1:11:1, 15:105:5]
    training_err = np.zeros(len(lam_vals))
    test_err = np.zeros(len(lam_vals))

    for i in range(len(lam_vals)):
        # Need to find w_hat,
        lam = lam_vals[i]
        training_err[i] = calc_err(lam, Xtrain_biased, ytrain, 'training')
        test_err[i] = calc_err(lam, Xtest_biased, ytest, 'test')

    plt.plot(lam_vals, training_err, label='training')
    plt.plot(lam_vals, test_err, label='test')
    plt.legend()
    plt.xlabel(r'$\lambda$')
    plt.ylabel("Error Rate")
    plt.title('Logistic regression')
    plt.savefig("q3.pdf", dpi=150)

    print()
