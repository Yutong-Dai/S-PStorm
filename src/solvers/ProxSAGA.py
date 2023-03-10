'''
# File: ProxSAGA.py
# Project: FaRSAGroup
# Created Date: 2022-09-19 9:43
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2022-11-28 11:35
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''


import numpy as np
import datetime
import time
import os
import sys
from copy import deepcopy
from scipy.sparse import csr_matrix
sys.path.append("../../")
import src.utils as utils
from src.solvers.BaseSolver import StoBaseSolver


class ProxSAGA(StoBaseSolver):
    def __init__(self, f, r, config):
        self.stepsize_strategy = config.proxsaga_stepsize
        self.version = "0.1 (2022-09-01)"
        self.solver = "ProxSAGA"
        super().__init__(f, r, config)

    def solve(self, x_init=None, alpha_init=None):
        # process argument
        if x_init is None:
            xk = np.zeros((self.p, 1))
        else:
            xk = x_init

        if not alpha_init:
            self.alphak = 1.0
        else:
            self.alphak = alpha_init

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

        self.compute_id_quantity = False

        # start computing

        while True:
            self.time_so_far = time.time() - start

            signal, gradfxk = self.check_termination(xk)
            if self.iteration == 0:
                dk = gradfxk
                # form the big gradient table; each column is a gradient component
                temp, grad_table = self.f.gradient(xk, idx=None, return_table=True)
                if grad_table is None:
                    print("Cannot store the grad_table due to the memory limit.")
                    return
                grad_table_mean = np.mean(grad_table, axis=1, keepdims=True)
                self.num_data_pass += 1
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
                # for the full epoch
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
                gradfxk_minibacth, gradfxk_minibacth_table = self.f.gradient(xk, idx=minibatch_idx, return_table=True)
                dk = (gradfxk_minibacth - np.mean(grad_table[:, minibatch_idx], axis=1, keepdims=True)) + grad_table_mean
                # update gradtable_mean
                grad_table_mean = (grad_table_mean * self.n - np.sum(grad_table[:, minibatch_idx], keepdims=True,
                                   axis=1) + np.sum(gradfxk_minibacth_table, axis=1, keepdims=True)) / self.n
                # update gradtable
                grad_table[:, minibatch_idx] = gradfxk_minibacth_table

                xkp1, _, _ = self.r.compute_proximal_gradient_update(xk, self.alphak, dk)
                self.iteration += 1
                xk = deepcopy(xkp1)

        # return solutions
        self.xend = xk
        self.Fend = self.Fxk

        self.print_exit()
        return self.collect_info(xk, F_seq, nz_seq, grad_error_seq, x_seq)

    def print_header(self):
        header = " Epoch.   Obj.    alphak      #z   #nz   |egradf| |   optim     #pz    #pnz |"
        header += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(header)
        else:
            print(header)

    def print_epoch(self):
        contents = f" {self.num_epochs:5d} {self.Fxk:.3e} {self.alphak:.3e} {self.nz:5d} {self.nnz:5d}  {self.grad_error:.3e} | {self.optim:.3e} {self.pz:5d}  {self.pnz:5d}  |"
        contents += "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)
