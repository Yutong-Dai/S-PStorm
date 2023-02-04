'''
# File: regularizer.py
# Project: ipg
# Created Date: 2021-08-23 11:31
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2023-02-03 9:03
# Modified By: Yutong Dai yutongdai95@gmail.com
#
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''

import numpy as np
from numba import jit
from scipy.sparse import csr_matrix


class GL1:
    def __init__(self, groups, penalty=None, weights=None):
        """
        !!Warning: need `groups` be ordered in a consecutive manner, i.e.,
        groups: array([1., 1., 1., 2., 2., 2., 3., 3., 3., 3.])
        Then:
        unique_groups: array([1., 2., 3.])
        group_frequency: array([3, 3, 4]))
        """
        self.penalty = penalty
        if penalty is None:
            raise ValueError("Initialization failed!")
        if type(groups) == np.ndarray:
            self.unique_groups, self.group_frequency = np.unique(
                groups, return_counts=True)
            if weights is not None:
                self.weights = weights
            else:
                self.weights = np.sqrt(self.group_frequency)
            self.weights = penalty * self.weights
            self.K = len(self.unique_groups)
            self.group_size = -1 * np.ones(self.K)
            p = groups.shape[0]
            full_index = np.arange(p)
            starts = []
            ends = []
            for i in range(self.K):
                G_i = full_index[np.where(groups == self.unique_groups[i])]
                # record the `start` and `end` indices of the group G_i to avoid fancy indexing innumpy
                # in the example above, the start index and end index for G_1 is 0 and 2 respectively
                # since python `start:end` will include `start` and exclude `end`, so we will add 1 to the `end`
                # so the G_i-th block of X is indexed by X[start:end]
                start, end = min(G_i), max(G_i) + 1
                starts.append(start)
                ends.append(end)
                self.group_size[i] = end - start
            # wrap as np.array for jit compile purpose
            self.starts = np.array(starts)
            self.ends = np.array(ends)
        else:
            print("Use fast initialization for the GL1 object", flush=True)
            if weights is not None:
                self.weights = weights
            else:
                self.weights = np.sqrt(groups['group_frequency'])
            self.weights = penalty * self.weights
            self.K = len(groups['group_frequency'])
            self.group_size = -1 * np.ones(self.K)
            self.starts = groups['starts']
            self.ends = groups['ends']
            self.unique_groups = np.linspace(1, self.K, self.K)

    def __str__(self):
        return("Group L1")

    def func(self, X):
        """
            X here is not the data matrix but the variable instead
        """
        return self._func_jit(X, self.K, self.starts, self.ends, self.weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _func_jit(X, K, starts, ends, weights):
        fval = 0.0
        for i in range(K):
            start, end = starts[i], ends[i]
            XG_i = X[start:end]
            # don't call l2_norm for jit to complie
            fval += weights[i] * np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
        return fval

    def grad(self, X):
        """
            compute the gradient. If evaluate at the group whose value is 0, then
            return np.inf for that group
        """
        return self._grad_jit(X, self.K, self.starts, self.ends, self.weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _grad_jit(X, K, starts, ends, weights):
        grad = np.full(X.shape, np.inf)
        for i in range(K):
            start, end = starts[i], ends[i]
            XG_i = X[start:end]
            norm_XG_i = np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
            if (np.abs(norm_XG_i) > 1e-15):
                grad[start:end] = (weights[i] / norm_XG_i) * XG_i
        return grad

    def _prepare_hv_data(self, X, subgroup_index):
        """
        make sure the groups in subgroup_index are non-zero
        """
        self.hv_data = {}
        start = 0
        for i in subgroup_index:
            start_x, end_x = self.starts[i], self.ends[i]
            XG_i = X[start_x:end_x]
            XG_i_norm = np.sqrt(np.dot(XG_i.T, XG_i))[0][0]
            end = start + end_x - start_x
            self.hv_data[i] = {}
            self.hv_data[i]['XG_i'] = XG_i
            self.hv_data[i]['XG_i_norm'] = XG_i_norm
            self.hv_data[i]['start'] = start
            self.hv_data[i]['end'] = end
            self.hv_data[i]['XG_i_norm_cubic'] = XG_i_norm**3
            start = end

    def hessian_vector_product_fast(self, v, subgroup_index):
        """
        call _prepare_hv_data before call hessian_vector_product_fast
        """
        hv = np.empty_like(v)
        for i in subgroup_index:
            start = self.hv_data[i]['start']
            end = self.hv_data[i]['end']
            vi = v[start:end]
            temp = np.matmul(self.hv_data[i]['XG_i'].T, vi)
            hv[start:end] = self.weights[i] * (1 / self.hv_data[i]['XG_i_norm'] * vi -
                                               (temp / self.hv_data[i]['XG_i_norm_cubic']) *
                                               self.hv_data[i]['XG_i'])
        return hv

    def _dual_norm(self, y):
        """
            compute the dual of r(x), which is r(y): max ||y_g||/lambda_g
            reference: https://jmlr.org/papers/volume18/16-577/16-577.pdf section 5.2
        """
        return self._dual_norm_jit(y, self.K, self.starts, self.ends, self.weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _dual_norm_jit(y, K, starts, ends, weights):
        max_group_norm = 0.0
        for i in range(K):
            start, end = starts[i], ends[i]
            yG_i = y[start:end]
            temp_i = (np.sqrt(np.dot(yG_i.T, yG_i))[0][0]) / weights[i]
            max_group_norm = max(max_group_norm, temp_i)
        return max_group_norm

    ##############################################
    #      exact proximal gradient calculation   #
    ##############################################
    def compute_proximal_gradient_update(self, xk, alphak, gradfxk):
        return self._compute_proximal_gradient_update_jit(xk, alphak, gradfxk, self.starts,
                                                          self.ends, self.weights)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _compute_proximal_gradient_update_jit(X, alpha, gradf, starts, ends, weights):
        proximal = np.zeros_like(X)
        nonZeroGroup = []
        zeroGroup = []
        for i in range(len(starts)):
            start, end = starts[i], ends[i]
            XG_i = X[start:end]
            gradfG_i = gradf[start:end]
            gradient_step = XG_i - alpha * gradfG_i
            gradient_step_norm = np.sqrt(
                np.dot(gradient_step.T, gradient_step))[0][0]
            if gradient_step_norm != 0:
                temp = 1 - ((weights[i] * alpha) / gradient_step_norm)
            else:
                temp = -1
            if temp > 0:
                nonZeroGroup.append(i)
            else:
                zeroGroup.append(i)
            proximal[start:end] = max(temp, 0) * gradient_step
        return proximal, len(zeroGroup), len(nonZeroGroup)

    def _get_group_structure(self, X):
        return self._get_group_structure_jit(X, self.K, self.starts, self.ends)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _get_group_structure_jit(X, K, starts, ends):
        nz = 0
        for i in range(K):
            start, end = starts[i], ends[i]
            X_Gi = X[start:end]
            if (np.sum(np.abs(X_Gi)) == 0):
                nz += 1
        nnz = K - nz
        return nnz, nz
