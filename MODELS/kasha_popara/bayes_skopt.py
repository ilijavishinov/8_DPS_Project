
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from skopt import BayesSearchCV

SEGMENTS_LENGTH = 3
EXPERIMENT_ID = F'BayesianSearch_{SEGMENTS_LENGTH}s'

data_dir = '/DATA/MODELS_DATA/DS1'
X_train = pd.read_csv(f'{data_dir}\\segments_{SEGMENTS_LENGTH}s_train.csv')
X_test = pd.read_csv(f'{data_dir}\\segments_{SEGMENTS_LENGTH}s_test.csv')

y_train = X_train.pop('episode')
y_test = X_test.pop('episode')

results = list()
workflow_dict = dict()

opt = BayesSearchCV(
    SVC(),
    {
        'C': (1e-6, 1e+6, 'log-uniform'),
        'gamma': (1e-6, 1e+1, 'log-uniform'),
        'degree': (1, 8),  # integer valued parameter
        'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    },
    n_iter=2,
    cv=3,
    verbose = 10,
)

opt.fit(X_train, y_train)
y_pred = opt.best_estimator_.predict(X_test)z
