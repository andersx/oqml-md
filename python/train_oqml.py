#!/usr/bin/env python3

import sys
import numpy as np
import pickle

from utils import gen_representations
from utils import train_alphas
from utils import QMLCalculatorOQML


if __name__ == "__main__":

    # NPZ filename
    data = np.load(sys.argv[1])

    # .dat filename for indexes
    train_idx = np.asarray(sorted(np.loadtxt(sys.argv[2], dtype=int)))

    # Kernel width, as default use 2.0
    sigma = float(sys.argv[3])

    # Regularizer for SVD, as default use 1e-11
    llambda = float(sys.argv[4])

    print(sys.argv)

    reps, dreps, nuclear_charges, energies = gen_representations(data)

    # Set model parameters
    parameters = {
        "sigma": sigma,     # kernel widht
        "offset": 0.0,      # Energy offset
        "llambda": llambda, # L2 regularizer
        }

    # Appropriately covert energies
    E = data["E"] # * 23.06035 # * 627.51
    F = data["F"] # * 23.06035 # * 627.51 # / 0.52917721092

    # Train model
    alphas = train_alphas(reps, dreps, nuclear_charges, E, F, train_idx, parameters)

    training_reps = reps[train_idx]
    training_dreps = dreps[train_idx]
    training_nuclear_charges  = [nuclear_charges[i] for i in train_idx]

    # Make ASE calculator
    calculator = QMLCalculatorOQML(parameters, training_reps, training_dreps, training_nuclear_charges, alphas)

    with open('md_model.pickle', 'wb') as handle:
        pickle.dump(calculator, handle, protocol=pickle.HIGHEST_PROTOCOL)
