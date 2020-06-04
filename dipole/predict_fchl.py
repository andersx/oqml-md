#!/home/andersx/opt/anaconda3/bin/python3

import sys
import numpy as np
from copy import deepcopy

from time import time

from tqdm import tqdm

from qml.fchl import generate_representation_electric_field
from qml.fchl import get_atomic_local_electric_field_gradient_kernels

DEBYE_TO_AU = 0.393456
DEBYE_TO_EAA = 0.20819434
# HARTREE_TO_KCAL_MOL = 1.0 #2.0# 627.509474
HARTREE_TO_KCAL_MOL = 627.509474
KCAL_MOL_TO_EV = 1.0 / 23.06035

ENERGY_UNIT = KCAL_MOL_TO_EV
DIPOLE_UNIT = DEBYE_TO_EAA

def xyz2npy(filename):

    f = open(sys.argv[1], "r")
    lines = f.readlines()
    f.close()

    mol_len = 4

    n_mols = len(lines) // (mol_len+2)
    
    coords = []

    for i in range(n_mols):

        idx_start = i * (mol_len+2)

        r = []
        for j in range(mol_len):

            l = idx_start + j + 2

            tokens = lines[l].split()

            x = float(tokens[1])
            y = float(tokens[2])
            z = float(tokens[3])

            r.append([x,y,z])
        coords.append(r)


    coords = np.array(coords)

    return coords



if __name__ == "__main__":
    
    
    Rall = xyz2npy(sys.argv[1])
    sigmas = [float(sys.argv[2])]

    df = 1e-3
    ef_scaling = 0.01

    kernel_args = {
        "kernel": "gaussian",
        "kernel_args": {
            "sigma": sigmas,
        },
        "cut_distance": 1e6,
        "alchemy": "off",
    }


    X = np.load("X.npy")
    alpha = np.load("alphas.npy")
    n_mols = Rall.shape[0]


    Xs = []

    print("Generating representations for trajectory ...")

    for i in tqdm(range(n_mols)):

        rep = generate_representation_electric_field(Rall[i], [6, 8, 1, 1], 
            fictitious_charges='gasteiger', max_size=4, cut_distance=1e6)

        Xs.append(rep)

    Xs = np.array(Xs)


    t_start = time()
    dKs = get_atomic_local_electric_field_gradient_kernels(X, Xs, df=df, ef_scaling=ef_scaling, **kernel_args)[0]
    t_end = time()
    print("Elapsed:", t_end - t_start)

    Ds = np.dot(dKs.T, alpha).reshape(-1,3)

    print("Ds")

    np.savetxt("dipole_fchl.dat", Ds)


