#!/usr/bin/env python3

import sys

from copy import deepcopy
import os
import numpy as np

from tqdm import tqdm

import qml
from qml.math import cho_solve
from qml.math import svd_solve
from qml.fchl import generate_representation_electric_field
from qml.fchl import generate_displaced_representations

from qml.fchl import get_local_symmetric_kernels
from qml.fchl import get_local_kernels

from qml.fchl import get_kernels_ef_field
from qml.fchl import get_atomic_local_electric_field_gradient_kernels

from qml.fchl import get_atomic_local_gradient_kernels
from qml.fchl import get_atomic_local_kernels 

from time import time

np.set_printoptions(linewidth=666)

DEBYE_TO_AU = 0.393456
DEBYE_TO_EAA = 0.20819434
# HARTREE_TO_KCAL_MOL = 1.0 #2.0# 627.509474
HARTREE_TO_KCAL_MOL = 627.509474
KCAL_MOL_TO_EV = 1.0 / 23.06035

ENERGY_UNIT = KCAL_MOL_TO_EV
DIPOLE_UNIT = DEBYE_TO_EAA


if __name__ == "__main__":

    print(sys.argv)

    data = np.load(sys.argv[1])
    n_mols = len(data["E"])

    train_idx = sorted(np.loadtxt(sys.argv[2], dtype=int))
    test_idx = np.array([i for i in range(n_mols) if i not in train_idx])

    
    sigmas = [float(sys.argv[3])]
    llambda = float(sys.argv[4])

    ef_scaling = 0.01
    df = 1e-3
    DX = 0.01
    kernel_args = {
        "kernel": "gaussian",
        "kernel_args": {
            "sigma": sigmas,
        },
        "cut_distance": 1e6,
        "alchemy": "off",
    }

    print(train_idx)
    print(test_idx)

    print(list(data.keys()))


    Dall = data["D"]
    Zall = data["Z"]
    Rall = data["R"]

    Xall = []

    # help(generate_representation_electric_field)

    for i in tqdm(range(n_mols)):

        rep  = generate_representation_electric_field(Rall[i], Zall[i], 
                fictitious_charges='gasteiger', max_size=4, cut_distance=1e6)

        Xall.append(rep)
    
    Xall = np.array(Xall)
    
    # Calculate kernels:

    X  = Xall[train_idx]
    Xs = Xall[test_idx]

    np.save("X.npy", X)

    D  = Dall[train_idx]
    Ds = Dall[test_idx]

    t_start = time()
    dK  = get_atomic_local_electric_field_gradient_kernels(X,  X, df=df, ef_scaling=ef_scaling, **kernel_args)[0]
    t_end = time()
    print("Elapsed:", t_start - t_end)
    
    t_start = time()
    dKs = get_atomic_local_electric_field_gradient_kernels(X, Xs, df=df, ef_scaling=ef_scaling, **kernel_args)[0]
    t_end = time()
    print("Elapsed:", t_start - t_end)

    t_start = time()

    Y = deepcopy(D.flatten())
    C = deepcopy(dK.T)

    dY = D.flatten()
    dYs = Ds.flatten()
        
    alpha = svd_solve(C, Y, rcond=llambda)
    np.save("alphas.npy", alpha)
    
    t_end = time()
    
    dYss = np.dot(dKs.T, alpha)
    dmae = np.mean(np.abs(dYs - dYss)) / DIPOLE_UNIT

    t_elapsed = t_end - t_start

    print("%7.2f    %20.12f Debye   %10.2f s" % (sigmas[0], dmae, t_elapsed))
