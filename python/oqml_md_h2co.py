#!/usr/bin/env python3

import sys

# sys.path.insert(0, "/home/andersx/dev/qml/gradient_kernel/build/lib.linux-x86_64-3.6")

from copy import deepcopy
import numpy as np

import scipy.linalg
from scipy.linalg import norm

from ase import Atoms
from ase.io import read, write
from ase.optimize import *
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from os.path import splitext
from ase.visualize import view

# from ase_util import transform_to_eckart_frame
import uuid

import ase
from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations

import qml
from qml.representations import generate_fchl_acsf
from qml.math import cho_solve
from qml.math import svd_solve

from tqdm import tqdm

import numpy as np
import time

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from qml.kernels import get_gp_kernel
from qml.kernels import get_symmetric_gp_kernel
from qml.kernels import get_atomic_local_kernel
from qml.kernels import get_atomic_local_gradient_kernel
from qml.representations import generate_fchl_acsf


new_cut = 8.0

cut_parameters = {
        "rcut": new_cut,
        "acut": new_cut,
        "nRs2": int(24 * new_cut / 8.0),
        "nRs3": int(20 * new_cut / 8.0),
}

def gen_representations(data):

    nuclear_charges = []


    print(list(data.keys()))

    print(data["Z"])


    max_atoms = 4
    elements = [1,6,8]

    print("max_atoms", max_atoms)
    print("elements", elements)

    
   
    reps = []
    dreps = []

    
    for i in tqdm(range(len(data["E"]))):
        x, dx = generate_fchl_acsf(data["Z"][i], data["R"][i], elements=elements, gradients=True,
                **cut_parameters)

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
            # pad=self.max_atoms,
            **cut_parameters)
        
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

    natoms = 4
    nmols = 4001
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
    Q  = nuclear_charges[train_idx]
    Qs = nuclear_charges[test_idx]

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


if __name__ == "__main__":

    temperature = 300
    timestep = 0.5
    friction = 0.01

    data = np.load(sys.argv[1])
    train_idx = sorted(np.loadtxt(sys.argv[2], dtype=int))
    sigma = float(sys.argv[3])
    llambda = float(sys.argv[4])

    E = data["E"] # * 23.06035 # * 627.51
    F = data["F"] # * 23.06035 # * 627.51 # / 0.52917721092

    reps = np.load("../learning_curves_new_splits/npy_data/X.npy")
    dreps = np.load("../learning_curves_new_splits/npy_data/dX.npy")
    
    # reps, dreps, nuclear_charges, energies = gen_representations(data)
    nuclear_charges = np.load("../learning_curves_new_splits/npy_data/Q.npy")

    parameters = {
        "sigma": sigma,
        "offset": 0.0,
        "llambda": llambda,
        }

    alphas = train_alphas(reps, dreps, nuclear_charges, E, F, train_idx, parameters)

    np.save("alphas.npy", alphas)
    # alphas = np.load("npy_data/alphas_gpr_%f.npy" % parameters["sigma"])

    calculator = QMLCalculatorOQML(parameters, reps[train_idx], dreps[train_idx], nuclear_charges[train_idx], 
            alphas)

    coordinates = np.array([
    [ 0.20123873, -0.39602112, -0.32005839],
    [-0.23296217,  0.48683071,  0.37425374],
    [ 0.6071696 , -0.20890494, -1.3476949 ],
    [ 0.22926032, -1.46543064,  0.01371154]])


    nuclear_charges = np.array([6, 8, 1, 1,])

    molecule = ase.Atoms(nuclear_charges, coordinates)
    molecule.set_calculator(calculator)

    BFGS(molecule).run(fmax=0.00001)

    vib = Vibrations(molecule, nfree=4)
    vib.run()
    vib.summary()

    # Set the momenta corresponding to a temperature T
    MaxwellBoltzmannDistribution(molecule, temperature * units.kB)

    # define the algorithm for MD: here Langevin alg. with with a time step of 0.1 fs,
    # the temperature T and the friction coefficient to 0.02 atomic units.
    dyn = Langevin(molecule, timestep * units.fs, temperature * units.kB, friction)


    R_list = []
    P_list = []


    t = time.time()
    # Equilibrate in NVT ensemble for 100'000 steps (=50ps)
    # for i in range(5000):
    for i in range(100000):

        dyn.run(1)
        if i % 100 == 0:
            print("NVT equilibration at step: " + str(i))
            print(time.time() - t, "s")
            t = time.time()
            
            epot = molecule.get_potential_energy()
            ekin = molecule.get_kinetic_energy()
            etot = epot + ekin
            print(etot)
            
            pos = molecule.get_positions()
            mom = molecule.get_momenta()
 
            R_list.append(pos)
            P_list.append(mom)

    R_list = np.array(R_list)
    P_list = np.array(R_list)

    np.save("pos_oqml_equil.npy", R_list)
    np.save("mom_oqml_equil.npy", P_list)


    #change to NVE ensemble
    dyn = VelocityVerlet(molecule, timestep * units.fs) 

    R_list = []
    P_list = []
    E_list = []
    F_list = []


    t = time.time()
    # for i in range(400000):
    for i in range(400000):
        dyn.run(1)
        
        if i % 100 == 0:
            print("NVE simulation, at step: " + str(i))
            print(time.time() - t, "s")
            t = time.time()

            epot = molecule.get_potential_energy()
            ekin = molecule.get_kinetic_energy()
            etot = epot + ekin
            print(etot)
 
        # pos = atoms.get_positions()
        # mom = atoms.get_momenta()
        pos = molecule.get_positions()
        mom = molecule.get_momenta()
 
        R_list.append(pos)
        P_list.append(mom)

    R_list = np.array(R_list)
    P_list = np.array(R_list)

    np.save("pos_oqml.npy", R_list)
    np.save("mom_oqml.npy", P_list)
