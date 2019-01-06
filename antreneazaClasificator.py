#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 23:46:31 2018

@author: gabriel
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
import pickle
import os.path
from time import time


def antreneazaClasificator(X_train, y_train, kernel):

    if kernel == 'liniar':

        model = LinearSVC()

        param_grid = [{'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]}]

        grid_search = gridSearch(X_train, y_train, model, param_grid)

        bestSVC = grid_search.best_estimator_

        w = bestSVC.coef_[0]
        b = bestSVC.intercept_

        scoruriExemple = np.sum(w * X_train, axis=1) + b

        scoruriExempleNegative = scoruriExemple[np.nonzero(y_train < 0)]
        scoruriExemplePozitive = scoruriExemple[np.nonzero(y_train > 0)]

        scExPoz, = plt.plot(np.sort(scoruriExemplePozitive), 'g')
        scExNeg, = plt.plot(np.sort(scoruriExempleNegative), 'r')
        plt.plot(np.array([0, max(scoruriExemplePozitive.shape[0],
                                  scoruriExempleNegative.shape[0])]),
                 np.array([0, 0]), 'b')
        plt.legend([scExPoz, scExNeg],
                   ['Scoruri exemple pozitive', 'Scoruri exemple negative'])
        plt.xlabel('nr. exemple de antrenare')
        plt.ylabel('scor clasificator')
        plt.title('Distributia scorurilor clasificatorului'
                  + ' pe exemplele de antrenare')
        plt.show()

        return w, b, grid_search.best_estimator_

    elif kernel == 'rbf':

        model = SVC()

        param_grid = [{'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]}]

        grid_search = gridSearch(X_train, y_train, model, param_grid)

        scoruriExemple = grid_search.predict(X_train)

        return None, None, grid_search.best_estimator_


def gridSearch(X_train, y_train, model, param_grid):

    start = time()
    numeClasificator = 'grid_searched_SVC.sav'

    # daca modelul a fost deja antrenat
    if os.path.isfile(numeClasificator):
        # incarca modelul
        grid_search = pickle.load(open(numeClasificator, 'rb'))
        print('Am incarcat modelul')

        stop = time() - start

    else:
        # antreneaza modelul
        grid_search = GridSearchCV(model, param_grid, cv=3,
                                   error_score=np.nan, n_jobs=1, verbose=0)

        grid_search.fit(X_train, y_train)

        stop = time() - start

        # salveaza modelul
        pickle.dump(grid_search, open(numeClasificator, 'wb'))
        print('Am salvat modelul')

    report(grid_search.cv_results_, stop)
    print(grid_search.score(X_train, y_train))
    return grid_search


def report(results, stop, n_top=1):
    """Utility function to report best scores"""
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("\nMean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

    print("GridSearch took %.2f seconds." % stop)
