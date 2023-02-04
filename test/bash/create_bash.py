'''
# File: create_bash.py
# Project: bash
# Created Date: 2022-09-15 11:57
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2023-02-04 2:35
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''

import os
import glob
import numpy as np
import sys
sys.path.append("../../")
import src.utils as utils


def create(scriptdir,
           purpose, data_dir, datasets, loss, lam_shrink, frac, weight_decay,
           accuracy, max_epochs, max_time, save_seq, seed, runs,
           solver, **kwargs):
    PROJ_DIR = '/home/yutong/S-PStorm'
    PYTHON_PATH = '/home/yutong/anaconda3/bin/python'           
    ext = ''
    if 'ext' in kwargs:
        ext = '_' + kwargs['ext']
    task_name = f'{solver}_{loss}_{lam_shrink}_{frac}{ext}'
    contents = f'cd {PROJ_DIR}/{scriptdir}\n\n'
    if solver != "FaRSAGroup":
        command = f"{PYTHON_PATH} main.py --purpose {purpose} --loss {loss} --lam_shrink {lam_shrink} --frac {frac} --weight_decay {weight_decay} --accuracy {accuracy} --max_epochs {max_epochs} --max_time {max_time} --save_seq {save_seq} --seed {seed} --runs {runs}"
    else:
        command = f"{PYTHON_PATH} main.py --purpose {purpose} --loss {loss} --lam_shrink {lam_shrink} --frac {frac} --weight_decay {weight_decay}"
    for data in datasets:
        task = command + f' --data_dir {data_dir} --dataset {data} --solver {solver}'
        task_hypers = ""
        for k, v in kwargs.items():
            if k != 'ext':
                task_hypers += f" --{k} {v}"
        task += task_hypers + f" >> ./bash/log/{task_name}.txt &"
        contents += task + '\n\n'
    filename = f'./{task_name}.sh'
    with open(filename, "w") as pbsfile:
        pbsfile.write(contents)


if __name__ == '__main__':
    # clean all existing bash files
    for f in glob.glob("*.sh"):
        os.remove(f)
    scriptdir = 'test'
    data_dir = '~/db'
    logdir = "./log"
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    to_dense = False
    node = '1'  # pick a random polyp node
    loss = 'logit'
    weight_decay = 1e-4
    accuracy = -1.0  # disable chi termination options
    ########################## config here ####################
    max_epochs = 1000
    max_time = 43200.0  # 12h
    save_seq = True
    seed = 2022
    runs = 1
    purpose = 'experiments'
    this_pbs_batch = 'final_run'
    dbsize = 'small'
    # criteria: N>1e4 and n>5e1
    if dbsize == 'small':
        datasets = ["a9a", 'covtype', "phishing", "w8a"]
    elif dbsize == 'medium':
        datasets = ['rcv1', 'real-sim']
    elif dbsize == 'large':
        datasets = ['avazu-app.tr', 'news20', 'url_combined', 'kdda']
    elif dbsize == 'find_a_better_machine':
        datasets = ['criteo', 'epsilon', 'webspam']
    else:
        raise ValueError(f"Incorrect dbsize:{dbsize}")

    ###### final run #########

    solver_hypers = {
        'FaRSAGroup': {},
        'ProxSVRG': {'proxsvrg_inner_repeat': 1},
        'ProxSAGA': {},
        'SPStorm': {'spstorm_stepsize': 'const', 'spstorm_betak': -1.0,
                    'spstorm_interpolate': True, 'spstorm_zeta': 'dynanmic'},
        'PStorm': {'pstorm_stepsize': 'diminishing', 'pstorm_betak': -1.0},
        'RDA': {'rda_stepsize': 'increasing', 'rda_stepconst': 1e-2}
    }
    count = 0
    for lam_shrink in [0.1, 0.01]:
        for frac in [0.25, 0.50, 0.75, 1.00]:
            for solver in ['ProxSVRG', 'ProxSAGA', 'RDA', 'PStorm', 'SPStorm']:
                # for solver in ['FaRSAGroup']:
                hypers = solver_hypers[solver]
                if dbsize != 'find_a_better_machine':
                    if dbsize != 'small' and solver == 'ProxSAGA':
                        continue
                    else:
                        create(scriptdir,
                               purpose, data_dir, datasets, loss, lam_shrink, frac, weight_decay,
                               accuracy, max_epochs, max_time, save_seq, seed, runs, solver, **hypers
                               )
