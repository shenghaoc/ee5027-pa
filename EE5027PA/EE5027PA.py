from bbnb import plot as bbnb_plot
from gnb import run as gnb_run
from knn import plot as knn_plot
from lr import plot as lr_plot
import matplotlib.pyplot as plt

plt.figure()
bbnb_plot()

gnb_run()

plt.figure()
lr_plot()

plt.figure()
knn_plot()
