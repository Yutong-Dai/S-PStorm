'''
File: params.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2019-10-31 15:51
Last Modified: 2021-02-01 00:29
--------------------------------------------
Description:
'''
from numpy import inf
params = {}
params["max_iter"] = 10000
params["max_time"] = 43200  # 12 hours
params["printlevel"] = 2
params["printevery"] = 20
params["Gamma"] = 1
params["eta_r"] = 1.0e-1
# try this no lager [1e-10, 1e-3]
params["eta"] = 1e-3
params["xi"] = 0.5
params["zeta"] = 0.8
params["kappa"] = 1  # 1e-5
params["tol_cg"] = 1.0e-6
params["tol_pg"] = 1.0e-6
params["betaFrac"] = 1
params["phiFrac"] = 1
params["tryCG"] = True
params["maxCG_iter"] = inf
params["tryCD"] = False
params["maxCD_iter"] = inf
params["maxback"] = 100
params["kappa1_max"] = 1e6
params["kappa2_max"] = 1e5
params["kappa1_min"] = 1e-5
params["kappa2_min"] = 1e-6
params["kappa_increase"] = 10
params["kappa_decrease"] = 0.1
params["kappa_freq_count"] = False
