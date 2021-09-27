# Q4. K-Nearest Neighbors
import numpy as np
from numpy import linalg as LA
import scipy.io as sio
import matplotlib.pyplot as plt

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

dist_arr_train = LA.norm(Xtrain[:, np.newaxis] - Xtrain, axis=2)
dist_arr_train = np.argsort(dist_arr_train)  # get the indices (cols) in sorted order
dist_arr_test = LA.norm(Xtest[:, np.newaxis] - Xtrain, axis=2)
dist_arr_test = np.argsort(dist_arr_test)  # get the indices (cols) in sorted order


def calc_err(k, dist_arr, y, mode):
    # Get knn indices then use them to access label
    knn_indices = dist_arr[:, :k]
    knn_labels = ytrain[knn_indices]

    # Original rows (X values we want to check) are preserved, so we apply
    # KNN formula to each row, but ignore k since its constant here
    result = np.less(np.sum(knn_labels == 0, axis=1), np.sum(knn_labels == 1, axis=1))
    err = np.sum(result.astype(int) != y.astype(int)) / y.size
    if k == 1 or k == 10 or k == 100:
        print('At k =', k, mode, 'error rate =', err)
    return err


def plot():
    print('Running KNN classifier')

    k_vals = np.r_[1:11:1, 15:105:5]

    training_err = np.zeros(len(k_vals))
    test_err = np.zeros(len(k_vals))

    for i in range(len(k_vals)):
        k = k_vals[i]
        training_err[i] = calc_err(k, dist_arr_train, ytrain, 'training')
        test_err[i] = calc_err(k, dist_arr_test, ytest, 'test')

    plt.plot(k_vals, training_err, label='training')
    plt.plot(k_vals, test_err, label='test')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel("Error Rate")
    plt.title('K-Nearest Neighbors')
    plt.savefig("q4.pdf", dpi=150)
