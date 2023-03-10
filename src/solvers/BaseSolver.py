'''
# File: BaseSolver.py
# Project: solvers
# Created Date: 2022-09-04 12:47
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2022-11-29 12:16
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
sys.path.append("../../")
import src.utils as utils
import copy
from scipy.sparse import csr_matrix


class StoBaseSolver:
    def __init__(self, f, r, config):
        self.f = f
        self.r = r
        self.n = self.f.n
        self.p = self.f.p
        self.K = self.r.K
        self.config = config
        self.full_idx = np.arange(self.n)
        self.datasetname = self.f.datasetName.split("/")[-1]
        if config.save_log:
            self.filename = '{}.txt'.format(self.config.tag)
        else:
            self.filename = None
        if self.config.print_level > 0:
            self.print_problem()
            self.print_config()
        # reproted stats
        self.status = 404
        self.num_epochs = 0
        self.num_data_pass = 0
        self.time_so_far = 0.0
        self.nnz, self.nz, self.nnz_best, self.nz_best = -1, -1, -1, -1
        self.Fbest = np.inf
        self.Fend = np.inf
        self.optim = np.inf
        self.xbest = None

    def print_problem(self):
        contents = "\n" + "=" * 80
        contents += f"\n       Solver:{self.solver}    Version: {self.version}  \n"
        contents += "=" * 80 + '\n'
        now = datetime.datetime.now()
        contents += f"Problem Summary: Excuted at {now.date()} {now.time()}\n"

        problem_attribute = self.f.__str__()
        problem_attribute += "Regularizer:{:.>56}\n".format(
            self.r.__str__())
        problem_attribute += "Penalty Parameter:{:.>30}lambda={:3.4f}\n".format(
            '', self.r.penalty)
        problem_attribute += "Number of groups:{:.>32}\n".format(self.r.K)
        contents += problem_attribute
        if self.filename is not None:
            with open(self.filename, "w") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def print_config(self):
        contents = "\n" + "Algorithm Parameters:\n"
        contents += 'Termination Conditions:'
        contents += f" accuracy: {self.config.accuracy} | optim scaled: {self.config.optim_scaled} | time limits:{self.config.max_time} | epoch limits:{self.config.max_iters}\n"
        contents += f"Sampling Setups: batchsize: {self.config.batchsize} | shuffle: {self.config.shuffle}\n"
        contents += f"Proximal Stepsize update: {self.stepsize_strategy}\n"
        contents += "*" * 100 + "\n"
        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def print_exit(self):
        contents = '\n' + "=" * 30 + '\n'
        if self.status == 0:
            contents += 'Exit: Optimal Solution Found\n'
        elif self.status == 1:
            contents += 'Exit: Iteration limit reached\n'
        elif self.status == 2:
            contents += 'Exit: Time limit reached\n'
        else:
            contents += f"Exits: {self.status} Something goes wrong"
        print(contents)
        contents += "\nFinal Results\n"
        contents += "=" * 30 + '\n'
        contents += f'Epochs:{self.num_epochs:d}\n'
        contents += f'# Data pass:{self.num_data_pass:d}\n'
        contents += f'CPU seconds:{self.time_so_far:.4f}\n'
        contents += f'# zero groups(best):{self.nz_best:d}\n'
        contents += f'# zero groups(end):{self.nz:d}\n'
        contents += f'Obj. F(best):{self.Fbest:8.6e}\n'
        contents += f'Obj. F(end):{self.Fend:8.6e}\n'
        contents += f'Optim.(end):{self.optim:8.6e}\n'

        if self.filename is not None:
            with open(self.filename, "a") as logfile:
                logfile.write(contents)
        else:
            print(contents)

    def solve(self):
        raise NotImplementedError("You shall not call this method!")

    def shuffleidx(self):
        if self.config.shuffle:
            bacthidx = np.random.permutation(self.full_idx)
        else:
            bacthidx = self.full_idx
        return bacthidx

    def collect_info(self, xk, F_seq, nz_seq, grad_error_seq, x_seq):
        info = {'num_epochs': self.num_epochs, 'num_data_pass': self.num_data_pass, 'time': self.time_so_far, 'optim': self.optim,
                'Fbest': self.Fbest, 'xbest': csr_matrix(self.xbest),
                'nnz_best': self.nnz_best, 'nz_best': self.nz_best,
                'Fend': self.Fxk, 'xend': csr_matrix(xk),
                'nnz': self.nnz, 'nz': self.nz,
                'status': self.status,
                'n': self.n, 'p': self.p, 'Lambda': self.r.penalty, 'K': self.r.K}
        if self.config.save_seq:
            info['F_seq'] = F_seq
            info['nz_seq'] = nz_seq
            info['grad_error_seq'] = grad_error_seq
        if self.config.save_xseq:
            info['x_seq'] = x_seq
        return info

    def check_termination(self, xk):
        # compute pg for check termination and get an idea of current progress
        # no need in real application
        # time_f_start = time.time()
        fxk = self.f.func(xk)
        # time_f = time.time() - time_f_start

        # time_r_start = time.time()
        rxk = self.r.func(xk)
        # time_r = time.time() - time_r_start

        self.Fxk = fxk + rxk

        # time_check_sparsity_start = time.time()
        self.nnz, self.nz = self.r._get_group_structure(xk)
        # time_check_sparsity = time.time() - time_check_sparsity_start

        if self.Fxk < self.Fbest:
            self.xbest = xk
            self.Fbest = self.Fxk
            self.nnz_best, self.nz_best = self.nnz, self.nz

        # time_grad_start = time.time()
        gradfxk = self.f.gradient(xk, idx=None)
        # time_grad = time.time() - time_grad_start

        # time_prox_start = time.time()
        xprox, self.pz, self.pnz = self.r.compute_proximal_gradient_update(xk, self.alphak, gradfxk)
        # time_prox = time.time() - time_prox_start

        # print(f"feval:{time_f:.1f} secs | reval:{time_r:.1f} secs | Feval:{time_f+time_r:.1f} secs | gradfeval:{time_grad:.1f} secs | proxeval:{time_prox:.1f} secs | time_check_sparsity: {time_check_sparsity:.1f} secs")

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
