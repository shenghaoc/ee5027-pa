# Q2. Gaussian Naive Bayes
import numpy as np
import scipy.io as sio

# 57 columns for X, 1 column for y
# 3065 for training, 1536 for test
# Need to squeeze out y from single-vaue ndarray
spam_contents = sio.loadmat('spamData.mat', squeeze_me=True)


# After filtering for value of y, use formula
# N_1/N, only care about heads
def gaussian_pdf(x, mu, sigma_2):
    return 1 / np.sqrt(2 * np.pi * sigma_2) * np.exp(-0.5 * (x - mu) ** 2 / sigma_2)


# Do preprocessing and use unqualified name for brevity
# Log-transform
Xtrain = np.log(np.add(spam_contents['Xtrain'], 0.1))
Xtest = np.log(np.add(spam_contents['Xtest'], 0.1))
ytrain = spam_contents['ytrain']
ytest = spam_contents['ytest']

# Use Strategy 1 (Maximum likelihood)
lambda_ml = np.sum(ytrain) / ytrain.size

mu_y_0 = np.apply_along_axis(np.mean, 0, Xtrain[ytrain == 0])
sigma_2_y_0 = np.apply_along_axis(np.var, 0, Xtrain[ytrain == 0])
mu_y_1 = np.apply_along_axis(np.mean, 0, Xtrain[ytrain == 1])
sigma_2_y_1 = np.apply_along_axis(np.var, 0, Xtrain[ytrain == 1])


def execute(Xsamp):
    # Calculate for class 0, y = 0
    # Within each class, need account for each feature individually
    log_p_y_0 = 1 - lambda_ml
    for i in range(len(Xsamp)):
        log_p_y_0 += np.log(gaussian_pdf(Xsamp[i], mu_y_0[i], sigma_2_y_0[i]))

    # Calculate for class 1, y = 1
    log_p_y_1 = lambda_ml
    for i in range(len(Xsamp)):
        log_p_y_1 += np.log(gaussian_pdf(Xsamp[i], mu_y_1[i], sigma_2_y_1[i]))

    return log_p_y_1 > log_p_y_0


def calc_err(X, y, mode):
    err_cnt = 0
    for v in range(len(y)):
        if execute(X[v]) != y[v]:
            err_cnt += 1

    err = err_cnt / len(y)
    print(mode, 'error rate =', err)


def run():
    print('Running Gaussian naive Bayes classifier')
    calc_err(Xtrain, ytrain, 'training')
    calc_err(Xtest, ytest, 'test')

    print()
