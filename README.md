# EE5907/EE5027 Programming Assignment

Please run `EE5027PA.py`, it will call the `plot` or `run` functions in the four separate Python files. Running these files themselves will not generate any output.

The file names are initials of the names of the classifiers.
1. `bbnb.py`, Beta-binomial Naive Bayes
2. `gnb.py`, Gaussian Naive Bayes
3. `lr.py`, Logistic regression
4. `knn.py`, K-Nearest Neighbors

For the Gaussian naive Bayes classifier, there is no variable to plot against, so it's invoked by `run` and not `plot`.

The plots will be saved as PDF files `q1.pdf`, `q3.pdf` and `q4.pdf` respectively. There will be no pop-up windows from Matplotlib.

The actual classification and calculation of error rates are handled by `calc_err` functions. For the first two questions, I looped over values of `ytrain` and `ytest` with an `execute` function, but classification was done directly for the other two questions with helper functions for Q4 closely mirroring the formulas given in the lecture notes.

Qualifiers for variable names were dropped for brevity but their full forms should be apparent with the knowledge of the specific type of classifier.
