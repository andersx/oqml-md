#!/usr/bin/env python3

import sys
import numpy as np
import time

from tqdm import tqdm
from copy import deepcopy

import qml
from qml.representations import generate_fchl_acsf
from qml.math import cho_solve
from qml.math import svd_solve
from qml.kernels import get_gp_kernel
from qml.kernels import get_symmetric_gp_kernel
from qml.kernels import get_atomic_local_kernel
from qml.kernels import get_atomic_local_gradient_kernel

import ase
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms

def gen_representations(data):

    nuclear_charges = []

    # print(list(data.keys()))

    # print(data["Z"])

    max_atoms = max([len(_) for _ in data["Z"]])
    elements = sorted(list(set(data["Z"].reshape(-1).tolist())))

    print("max_atoms", max_atoms)
    print("elements", elements)

    reps = []
    dreps = []
    
    for i in tqdm(range(len(data["E"]))):
        x, dx = generate_fchl_acsf(data["Z"][i], data["R"][i], 
                elements=elements, gradients=True,
                pad=max_atoms)

        reps.append(x)
        dreps.append(dx)

    energies        = data["E"].flatten()
    nuclear_charges = data["Z"].tolist()

    reps = np.array(reps)
    dreps = np.array(dreps)

    return reps, dreps, nuclear_charges, energies


class QMLCalculatorOQML(Calculator):
    name = 'QMLCalculator'
    implemented_properties = ['energy', 'forces']

    def __init__(self, parameters, representations, deriv_reps, charges, alphas, reducer=None, **kwargs):
        super().__init__(**kwargs)
        # unpack parameters
        offset = parameters["offset"]
        sigma = parameters["sigma"]

        self.set_model(alphas, representations, deriv_reps, charges, offset, sigma)
        self.energy = 0.0
        self.reducer = reducer

        self.last_atoms = -1

        if reducer is not None:
            self.repr = np.einsum("ijk,kl->ijl", representations,  reducer)
            self.drepr  = np.einsum("ijkmn,kl->ijlmn", deriv_reps,  reducer)
    
    
    def calculate(self, atoms: Atoms = None, properties=('energy', 'forces'),
                  system_changes=all_changes):

        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError(
                'No ASE atoms supplied to calculation, and no ASE atoms supplied with initialisation.')

        self.query(atoms)

        if 'energy' in properties:
            self.results['energy'] = self.energy

        if 'forces' in properties:
            self.results['forces'] = self.forces

        return

    def set_model(self, alphas, representations, deriv_reps, charges, offset, sigma):

        self.alphas = alphas
        self.repr = representations
        self.drepr = deriv_reps
        self.charges = charges

        # Offset from training
        self.offset = offset

        # Hyper-parameters
        self.sigma = sigma
        self.max_atoms = self.repr.shape[1]

        self.n_atoms = len(charges[0])

        return

    def query(self, atoms=None, print_time=True):

        if print_time:
            start = time.time()

        # kcal/mol til ev
        # kcal/mol/aangstrom til ev / aangstorm
        conv_energy = 1.0 #0.0433635093659
        conv_force  = 1.0 # 0.0433635093659

        coordinates = atoms.get_positions()
        nuclear_charges = atoms.get_atomic_numbers()
        n_atoms = coordinates.shape[0]


        rep_start = time.time()

        rep, drep = generate_fchl_acsf(
            nuclear_charges,
            coordinates,
            gradients=True,
            elements=[1,6,8],
            pad=self.max_atoms,
            )
        
        Qs = [nuclear_charges]
        Xs = np.array([rep], order="F")
        dXs = np.array([drep], order="F")
        
        if self.reducer is not None:
            Xs  = np.einsum("ijk,kl->ijl", Xs, self.reducer)
            dXs = np.einsum("ijkmn,kl->ijlmn", dXs, self.reducer)

        rep_end = time.time()


        kernel_start = time.time()
        # Ks = get_gp_kernel(self.repr, Xs, self.drepr, dXs, self.charges, Qs, self.sigma)
        
        Kse = get_atomic_local_kernel(self.repr, Xs, self.charges, Qs, self.sigma)
        Ksf = get_atomic_local_gradient_kernel(self.repr, Xs, dXs, self.charges, Qs, self.sigma)

        kernel_end = time.time()

        pred_start = time.time()
        # Energy prediction
        energy_predicted = np.dot(Kse, self.alphas)[0] + self.offset
        self.energy = energy_predicted * conv_energy

        # Force prediction
        forces_predicted = np.dot(Ksf, self.alphas).reshape((n_atoms, 3))
        self.forces = forces_predicted * conv_force

        pred_end = time.time()

        if print_time:
            end = time.time()
            # print("rep        ", rep_end - rep_start)
            # print("kernel     ", kernel_end - kernel_start)
            # print("prediciton ", pred_end - pred_start)
            # print("qml query {:7.3f}s {:10.3f} ".format(end-start, energy_predicted))


        return

    def calculation_required(atoms, quantities):

        print("TEST")

        if atoms == self.last_atoms:
            print("Not required")
            return False
        else:
            print("Required")
            return True

        

    def get_potential_energy(self, atoms=None, force_consistent=False):

        do_query = True
        
        try:
            D = atoms.get_positions()  - self.last_atoms
            if norm(D) < 1e-12:
                do_query = False
        except:
            do_query = True

        if do_query:
            self.last_atoms = atoms.get_positions()
            self.query(atoms=atoms)

        energy = self.energy

        return energy

    def get_forces(self, atoms=None):

        # print(atoms.get_positions())
        # print(self.last_atoms)

        do_query = True

        try:
            D = atoms.get_positions()  - self.last_atoms
            if norm(D) < 1e-12:
                do_query = False
        except:
            do_query = True

        if do_query:
            self.last_atoms = atoms.get_positions()
            self.query(atoms=atoms)
        
        forces = self.forces

        return forces

def train_alphas(reps, dreps, nuclear_charges, E, F, train_idx, parameters):

    print(reps.shape)

    all_idx = np.array(list(range(4001)))
    test_idx = np.array([i for i in all_idx if i not in train_idx])

    print(train_idx)
    print(test_idx)

    natoms = reps.shape[1]
    nmols = len(E)
    atoms = np.array([i for i in range(natoms*3)])

    train_idx_force = np.array([atoms + (3*natoms)*j + nmols for j in train_idx]).flatten()
    test_idx_force  = np.array([atoms + (3*natoms)*j + nmols for j in test_idx]).flatten()

    idx = np.concatenate((train_idx, train_idx_force))
    
    n_train = len(train_idx)
    n_test = len(test_idx)
    
    X  = reps[train_idx]
    Xs = reps[test_idx]
    dX  = dreps[train_idx]
    dXs = dreps[test_idx]
    Q  = [nuclear_charges[i] for i in train_idx]
    Qs = [nuclear_charges[i] for i in test_idx]

    Ke = get_atomic_local_kernel(X, X, Q, Q, parameters["sigma"])
    Kf = get_atomic_local_gradient_kernel(X, X, dX, Q, Q, parameters["sigma"])

    C = np.concatenate((Ke, Kf)) 

    Kes = get_atomic_local_kernel(X, Xs, Q, Qs, parameters["sigma"])
    Kfs = get_atomic_local_gradient_kernel(X, Xs, dXs, Q, Qs, parameters["sigma"])

    Y = np.concatenate((E[train_idx], F[train_idx].flatten()))

    alphas = svd_solve(C, Y, rcond=parameters["llambda"])

    eYs = deepcopy(E[test_idx])
    fYs = deepcopy(F[test_idx]).flatten()

    eYss = np.dot(Kes, alphas)
    fYss = np.dot(Kfs, alphas)

    ermse_test = np.sqrt(np.mean(np.square(eYss - eYs)))
    emae_test = np.mean(np.abs(eYss - eYs))

    frmse_test = np.sqrt(np.mean(np.square(fYss - fYs)))
    fmae_test = np.mean(np.abs(fYss - fYs))

    schnet_score  = 0.01 * sum(np.square(eYss - eYs))
    schnet_score += sum(np.square(fYss - fYs)) / natoms

    print("TEST  %5.2f  %.2E  %6.4e  %10.8f  %10.8f  %10.8f  %10.8f" % \
            (parameters["sigma"], parameters["llambda"], schnet_score, emae_test, ermse_test, fmae_test, frmse_test))
    
    return alphas
