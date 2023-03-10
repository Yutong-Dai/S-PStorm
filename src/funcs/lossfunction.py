'''
File: lossfunction.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2020-05-11 18:03
Last Modified: 2021-02-27 21:18
--------------------------------------------
Description:
'''
import numpy as np
from scipy import sparse
from numpy.linalg import eigvalsh
from numba import jit


class LogisticLoss:
    def __init__(self, X, y, datasetName=None, weight_decay=0.0):
        self.n, self.p = X.shape
        self.X, self.y = X, y
        self.X_batch, self.y_batch = None, None
        self.expterm_batch = None
        self.expterm = None
        self.sigmoid_batch = None
        self.sigmoid = None
        self.datasetName = datasetName
        self.weight_decay = weight_decay

    def __str__(self):
        info = ""
        if self.datasetName is not None:
            info += "Dataset:{:.>48}\n".format(self.datasetName)
        info += "Data Size:{:.>38}n={}, p={}\nLoss Function:{:.>34}Logistic\n".format(
            '', self.n, self.p, '')
        return info

    def _set_expterm(self, weight, bias=0, idx=None):
        if idx is not None:
            self.y_batch = self.y[idx, :]
            self.X_batch = self.X[idx, :]
            self.expterm_batch = np.exp(-(self.y_batch)
                                        * (self.X_batch @ weight + bias))
        else:
            self.expterm = np.exp(-(self.y) * (self.X @ weight + bias))

    def func(self, weight):
        """
        function value of logistic loss function evaluate at the given point weight.
        f(weight) = frac{1}{n} sum_{i=1}^n log(1+exp(y_i * weight^T * x_i))
        """
        f = self.evaluate_function_value(weight, bias=0, idx=None)
        return f

    def evaluate_function_value(self, weight, bias=0, idx=None):
        """
        function value of logistic loss function evaluate at the given point (weight, bias).
        f(weight,bias) = frac{1}{n} sum_{i=1}^n log(1+exp(y_i * weight^T * x_i))
        """
        self._set_expterm(weight, bias, idx)
        if idx is not None:
            f = np.sum(np.log(1 + self.expterm_batch)) / len(idx)
        else:
            f = np.sum(np.log(1 + self.expterm)) / self.n
        return f

    def _set_sigmoid(self, idx=None):
        if idx is not None:
            self.sigmoid_batch = 1 - (1 / (1 + self.expterm_batch))
        else:
            self.sigmoid = 1 - (1 / (1 + self.expterm))

    def gradient(self, xk, idx=None, return_table=False):
        """
        need to be called after `evaluate_function_value` to get correct `expterm`
        """
        self._set_sigmoid(idx)
        if return_table:
            if idx is not None:
                table = (-self.y_batch * self.sigmoid_batch) * self.X_batch.toarray()
            else:
                try:
                    table = (-self.sigmoid * self.y) * self.X.toarray()
                except Exception as e:
                    print(str(e))
                    return None, None
            table = table.T
            gradient = np.mean(table, axis=1, keepdims=True)
            return gradient + self.weight_decay * xk, table + self.weight_decay * xk
        else:
            if idx is not None:
                gradient = -((self.sigmoid_batch * self.y_batch).T @
                             self.X_batch) / len(idx)
            else:
                gradient = -((self.sigmoid * self.y).T @ self.X) / self.n

            return gradient.T + self.weight_decay * xk

    def _prepare_hv_data(self, subgroup_index):
        self.X_subset = self.X[:, subgroup_index]
        self.sigmoid_prod = (self.sigmoid * (1 - self.sigmoid))
        if self.weight_decay == 0.0:
            self.sigmoid_prod = np.maximum(1e-8, self.sigmoid_prod)

    def hessian_vector_product_fast(self, v):
        A = (self.X_subset @ v)
        B = A * self.sigmoid_prod
        hv = (B.T @ self.X_subset).T / self.n
        if self.weight_decay > 0.0:
            hv += self.weight_decay * v
        return hv


class LeastSquares:
    def __init__(self, X, y, datasetName=None, weight_decay=0.0):
        """
        docstring
        """
        self.n, self.p = X.shape
        self.X, self.y = X, y
        self.datasetName = datasetName
        self.weight_decay = weight_decay

    def __str__(self):
        info = ""
        if self.datasetName is not None:
            info += "Dataset:{:.>48}\n".format(self.datasetName)
        info += "Data Size:{:.>38}n={}, p={}\nLoss Function:{:.>34}LeastSquares\n".format(
            '', self.n, self.p, '')
        return info

    def evaluate_function_value(self, weight):
        """

        """
        self.matvec = self.X @ weight - self.y
        f = 0.5 * np.sum(self.matvec * self.matvec) / self.n
        self.weight = weight
        return f

    def gradient(self):
        """
        need to be called after `evaluate_function_value` to get correct `expterm`
        """
        gradient = self.matvec.T @ self.X / self.n
        return gradient.T + self.weight_decay * self.weight

    def _prepare_hv_data(self, subgroup_index):
        self.X_subset = self.X[:, subgroup_index]

    def hessian_vector_product_fast(self, v):
        temp = (self.X_subset @ v)
        hv = (temp.T @ self.X_subset).T / self.n
        return hv + 1e-8 * v

    def hessian(self):
        return self.X.T @ self.X / self.n
