#!/usr/bin/env python3

import sys
import pickle

from copy import deepcopy
import numpy as np

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

import ase
from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations

from tqdm import tqdm

import numpy as np
import time

if __name__ == "__main__":

    # MD settings
    temperature = 300
    timestep = 0.5
    friction = 0.01

    equilibration_steps = 100000
    production_steps = 400000


    with open('md_model.pickle', 'rb') as handle:
        calculator = pickle.load(handle)

    # Starting geometry
    coordinates = np.array([
        [ 0.20123873, -0.39602112, -0.32005839],
        [-0.23296217,  0.48683071,  0.37425374],
        [ 0.6071696 , -0.20890494, -1.3476949 ],
        [ 0.22926032, -1.46543064,  0.01371154]])

    nuclear_charges = np.array([6, 8, 1, 1,])

    molecule = ase.Atoms(nuclear_charges, coordinates)
    molecule.set_calculator(calculator)


    # Optimize initial geometry
    BFGS(molecule).run(fmax=0.00001)

    # vib = Vibrations(molecule, nfree=4)
    # vib.run()
    # vib.summary()

    # Set the momenta corresponding to a temperature T
    MaxwellBoltzmannDistribution(molecule, temperature * units.kB)

    # define the algorithm for MD: here Langevin alg. with with a time step of 0.1 fs,
    # the temperature T and the friction coefficient to 0.02 atomic units.
    dyn = Langevin(molecule, timestep * units.fs, temperature * units.kB, friction)


    R_list = []
    P_list = []


    t = time.time()
    for i in range(equilibration_steps):

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

    np.save("pos_fchl_oqml_equil.npy", R_list)
    np.save("mom_fchl_oqml_equil.npy", P_list)


    #change to NVE ensemble
    dyn = VelocityVerlet(molecule, timestep * units.fs) 

    R_list = []
    P_list = []
    E_list = []
    F_list = []


    t = time.time()
    for i in range(production_steps):
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

    np.save("pos_fchl_oqml.npy", R_list)
    np.save("mom_fchl_oqml.npy", P_list)
