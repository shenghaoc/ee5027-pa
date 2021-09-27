# Q1. Beta-binomial Naive Bayes
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# 57 columns for X, 1 column for y
# 3065 for training, 1536 for test
# Need to squeeze out y from single-vaue ndarray
spam_contents = sio.loadmat('spamData.mat', squeeze_me=True)


# After filtering for value of y, use formula
# (N_1+a)/(N+a+b), only care about heads
def p_x_1(col, a, b):
    return (np.count_nonzero(col) + a) / (col.size + a + b)


# Do preprocessing and use unqualified name for brevity
# Binarization
Xtrain = np.where(spam_contents['Xtrain'] > 0, 1, 0)
Xtest = np.where(spam_contents['Xtest'] > 0, 1, 0)
ytrain = spam_contents['ytrain']
ytest = spam_contents['ytest']

# Use strategy 3: posterior predictive
lambda_ml = np.sum(ytrain) / ytrain.size


# No further preprocessing, O(N!) space complexity to store all combinations of features


def execute(Xsamp, p_x_1_arr_y_0, p_x_1_arr_y_1):
    # Omitted tildes and conditions in names for brevity
    # Calculate for class 0, y = 0
    # Within each class, need account for each feature individually
    log_p_y_0 = np.log(1 - lambda_ml) + sum(np.log(np.where(Xsamp == 1, p_x_1_arr_y_0, 1 - p_x_1_arr_y_0)))

    # Calculate for class 1, y = 1
    log_p_y_1 = np.log(lambda_ml) + sum(np.log(np.where(Xsamp == 1, p_x_1_arr_y_1, 1 - p_x_1_arr_y_1)))

    return log_p_y_1 > log_p_y_0


def calc_err(a, p_x_1_arr_y_0, p_x_1_arr_y_1, X, y, mode):
    err_cnt = 0
    for v in range(len(y)):
        if execute(X[v], p_x_1_arr_y_0, p_x_1_arr_y_1) != y[v]:
            err_cnt += 1

    err = err_cnt / len(y)
    if a == 1 or a == 10 or a == 100:
        print('At alpha =', a, mode, 'error rate =', err)
    return err


def plot():
    print('Running Beta-Binomial naive Bayes classifier')
    a_vals = np.arange(0, 100.5, 0.5)

    training_err = np.zeros(len(a_vals))
    test_err = np.zeros(len(a_vals))

    for i in range(len(a_vals)):
        a = a_vals[i]

        p_x_1_arr_y_0 = np.apply_along_axis(p_x_1, 0, Xtrain[ytrain == 0], a, a)
        p_x_1_arr_y_1 = np.apply_along_axis(p_x_1, 0, Xtrain[ytrain == 1], a, a)

        training_err[i] = calc_err(a, p_x_1_arr_y_0, p_x_1_arr_y_1, Xtrain, ytrain, 'training')
        test_err[i] = calc_err(a, p_x_1_arr_y_0, p_x_1_arr_y_1, Xtest, ytest, 'test')

    plt.plot(a_vals, training_err, label='training')
    plt.plot(a_vals, test_err, label='test')
    plt.legend()
    plt.title('Beta-binomial Naive Bayes')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Error Rate')
    plt.savefig('q1.pdf', dpi=150)
