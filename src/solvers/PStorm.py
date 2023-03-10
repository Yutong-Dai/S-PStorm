'''
# File: solver.py
# Project: src
# Created Date: 2022-03-21 2:58
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2022-11-28 11:36
# Modified By: Yutong Dai yutongdai95@gmail.com
#
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
# nearly the same as the ProxStorm; just remove the interpolation step
'''

import numpy as np
import datetime
import time
import os
from copy import deepcopy
from scipy.sparse import csr_matrix
import sys
sys.path.append("../../")
import src.utils as utils
from src.solvers.BaseSolver import StoBaseSolver


class PStorm(StoBaseSolver):
    def __init__(self, f, r, config):
        self.stepsize_strategy = config.pstorm_stepsize
        self.version = "0.1 (2022-09-16)"
        self.solver = "Pstorm"
        super().__init__(f, r, config)

    def solve(self, x_init=None, alpha_init=None, Lg=None, **kwargs):
        # process argument
        if x_init is None:
            xk = np.zeros((self.p, 1))
        else:
            xk = x_init
        xkm1 = deepcopy(xk)
        yk = deepcopy(xk)

        if self.n % self.config.batchsize == 0:
            self.num_batches = self.n // self.config.batchsize
        else:
            self.num_batches = self.n // self.config.batchsize + 1

        # collect stats
        self.iteration = 0
        self.time_so_far = 0
        start = time.time()
        
        F_seq = []
        nz_seq = []
        grad_error_seq = []
        x_seq = []

        if self.stepsize_strategy == "const":
            self.alphak = alpha_init
            stepconst = np.power(4, 1 / 3) / (8 * Lg)
        else:
            # this choice is inspired by
            # Momentum-based variance-reduced proximal stochastic gradient method for composite nonconvex stochastic optimization
            # https://arxiv.org/abs/2006.00425
            # used in the non-convex setting
            stepconst = np.power(4, 1 / 3) / (8 * Lg)
            self.alphak = stepconst / np.power(self.iteration + 4, 1 / 3)
        self.betak = 1.0

        self.compute_id_quantity = False

        # start computing
        dk = 1e17
        while True:
            self.time_so_far = time.time() - start
            signal, gradfxk = self.check_termination(xk)

            self.grad_error = utils.l2_norm(dk - gradfxk)

            if self.config.save_seq:
                F_seq.append(self.Fxk)
                nz_seq.append(self.nz)
                grad_error_seq.append(self.grad_error)
            if self.config.save_xseq:
                x_seq.append(csr_matrix(xk))                
            if signal == "terminate":
                self.print_epoch()
                break
            else:
                self.num_data_pass += 1

            # print current epoch information
            if self.config.print_level > 0:
                if self.num_epochs % self.config.print_head_every == 0:
                    self.print_header()
                self.print_epoch()

            # new epoch
            self.num_epochs += 1
            batchidx = self.shuffleidx()
            for i in range(self.num_batches):
                start_idx = i * self.config.batchsize
                end_idx = min(start_idx + self.config.batchsize, self.n)
                minibatch_idx = batchidx[start_idx:end_idx]
                _ = self.f.evaluate_function_value(xk, bias=0, idx=minibatch_idx)
                vk = self.f.gradient(xk, idx=minibatch_idx)
                if self.iteration == 0:
                    dk = vk
                else:
                    _ = self.f.evaluate_function_value(xkm1, bias=0, idx=minibatch_idx)
                    uk = self.f.gradient(xkm1, idx=minibatch_idx)
                    if self.stepsize_strategy == "diminishing":
                        self.alphakp1 = stepconst / np.power(self.iteration + 4 + 1, 1 / 3)
                        self.betak = (1 + 24 * self.alphak**2 * Lg**2 - self.alphakp1 / self.alphak) / (1 + 4 * self.alphak**2 * Lg**2)
                    elif self.stepsize_strategy == "const":
                        if self.config.pstorm_betak == -1.0:
                            self.alphakp1 = stepconst / np.power(self.iteration + 5, 1 / 3)
                            self.betak = (1 + 24 * self.alphak**2 * Lg**2 - self.alphakp1 / self.alphak) / (1 + 4 * self.alphak**2 * Lg**2)
                        else:
                            self.betak = self.config.pstorm_betak
                        # raise ValueError(f"No theoretical guarantee for the stepsize strategy:{self.stepsize_strategy}")
                    else:
                        raise ValueError(f"Unknown stepsize strategy:{self.stepsize_strategy}")

                    dk = vk + (1 - self.betak) * (dkm1 - uk)
                # new iterate
                xkp1, _, _ = self.r.compute_proximal_gradient_update(xk, self.alphak, dk)
                self.iteration += 1

                # adjust stepsize
                if self.stepsize_strategy == "diminishing":
                    self.alphak = stepconst / np.power(self.iteration + 4, 1 / 3)

                dkm1 = deepcopy(dk)
                xkm1 = deepcopy(xk)
                xk = deepcopy(xkp1)

        # return solutions
        self.xend = xk
        self.Fend = self.Fxk
        self.print_exit()
        return self.collect_info(xk, F_seq, nz_seq, grad_error_seq, x_seq)

    def print_header(self):
        header = " Epoch.   Obj.    alphak     betak      #z   #nz   |egradf| |   optim     #pz    #pnz |"
        if self.compute_id_quantity:
            header += "  id_left    id_right  maxq<delta |"
        header += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(header)
        else:
            print(header)

    def print_epoch(self):
        contents = f" {self.num_epochs:5d} {self.Fxk:.3e} {self.alphak:.3e} {self.betak:.3e} {self.nz:5d} {self.nnz:5d}  {self.grad_error:.3e} | {self.optim:.3e} {self.pz:5d}  {self.pnz:5d}  |"
        if self.compute_id_quantity:
            contents += f" {self.id_left:.3e}  {self.id_rifgt:.3e}    {str(self.check):5s}    |"
        contents += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def check_termination_new(self, xk, yk):
        # override the default behavior
        fxk = self.f.func(xk)
        rxk = self.r.func(xk)
        self.Fxk = fxk + rxk
        self.nnz, self.nz = self.r._get_group_structure(yk)
        if self.Fxk < self.Fbest:
            self.xbest = xk
            self.Fbest = self.Fxk
            self.nnz_best, self.nz_best = self.nnz, self.nz
        gradfxk = self.f.gradient(xk, idx=None)
        xprox, self.pz, self.pnz = self.r.compute_proximal_gradient_update(xk, self.alphak, gradfxk)
        self.optim = utils.l2_norm(xprox - xk)
        if self.config.optim_scaled:
            self.optim = self.optim / max(1e-15, self.alphak)

        if self.optim <= self.config.accuracy:
            self.status = 0
            return 'terminate', gradfxk
        if self.num_epochs >= self.config.max_epochs:
            self.status = 1
            return 'terminate', gradfxk
        if self.time_so_far >= self.config.max_time:
            self.status = 2
            return 'terminate', gradfxk
        return 'continue', gradfxk
