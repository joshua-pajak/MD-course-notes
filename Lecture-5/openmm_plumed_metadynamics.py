import os
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmplumed import PlumedForce

# --- Input files ---
inpcrd = AmberInpcrdFile('../c148s_dimer.inpcrd')
prmtop = AmberPrmtopFile('../c148s_dimer.prmtop', periodicBoxVectors=inpcrd.boxVectors)

# --- System configuration ---
nonbondedMethod = PME
nonbondedCutoff = 1.0*nanometers
ewaldErrorTolerance = 0.0005
constraints = HBonds
rigidWater = True
constraintTolerance = 0.000001
hydrogenMass = 3*amu

# --- Integration options ---
dt = 0.004*picoseconds
temperature = 310*kelvin
friction = 1.0/picosecond
pressure = 1.0*atmospheres
barostatInterval = 25

# --- Equilibration simulation options ---
NPTsteps = 25000000
NVTsteps = 250000
platform = Platform.getPlatformByName('CUDA')
platformProperties = {'Precision': 'mixed'}
NVTdcdReporter = DCDReporter('NVT.dcd', 10000)
NPTdcdReporter = DCDReporter('NPT.dcd', 625000)
NPTdataReporter = StateDataReporter('npt.txt', 10000, totalSteps=NPTsteps,
   step=True, speed=True, progress=True, potentialEnergy=True, temperature=True, separator='\t')
NPTcheckpointReporter = CheckpointReporter('npt.chk', 100000)

# --- Prepare the Simulation ---
print('Building system...')
system = prmtop.createSystem(nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff, constraints=constraints, rigidWater=rigidWater, ewaldErrorTolerance=ewaldErrorTolerance, hydrogenMass=hydrogenMass)
integrator = LangevinMiddleIntegrator(temperature, friction, dt)
integrator.setConstraintTolerance(constraintTolerance)
simulation = Simulation(prmtop.topology, system, integrator, platform, platformProperties)
simulation.context.setPositions(inpcrd.positions)

# --- Minimize and Equilibrate ---
print('Performing energy minimization...')
simulation.minimizeEnergy()
print('Equilibrating NVT...')
simulation.context.setVelocitiesToTemperature(temperature)
simulation.reporters.append(NVTdcdReporter)
simulation.step(NVTsteps)

# --- Simulate NPT ---
print('Simulating NPT...')
simulation.reporters.clear()
system.addForce(MonteCarloBarostat(pressure, temperature, barostatInterval))
simulation.context.reinitialize(preserveState=True)
simulation.reporters.append(NPTdcdReporter)
simulation.reporters.append(NPTdataReporter)
simulation.reporters.append(NPTcheckpointReporter)
simulation.currentStep = 0
simulation.step(NPTsteps)

# --- Metadynamics production ---
'''
print('Saving template.pdb...')
modeller = Modeller(prmtop.topology, inpcrd.positions)
modeller.deleteWater()
PDBFile.writeFile(modeller.topology, modeller.positions, open('template.pdb', 'w'))
'''

print('Starting PLUMED metadynamics...')
plumed_script = f"""
MOLINFO STRUCTURE=../template.pdb MOLTYPE=protein
# define groups for distance (backbone N CA C O)
com1: COM ATOMS={{@mdt:{{resid 133 to 251 and backbone}}}}
com2: COM ATOMS={{@mdt:{{resid 261 to 376 and backbone}}}}

# define collective variables
cv_helix: ALPHARMSD RESIDUES=404-414
cv_dist: DISTANCE ATOMS=com1,com2 

uwall: UPPER_WALLS ARG=cv_dist AT=9.5 KAPPA=150.0 EXP=2 EPS=1 OFFSET=0
lwall: LOWER_WALLS ARG=cv_dist AT=4.2 KAPPA=150.0 EXP=2 EPS=1 OFFSET=0

# Metadynamics settings
METAD ...
	ARG=cv_helix,cv_dist
	SIGMA=0.1,0.2
	HEIGHT=0.2
	PACE=500
	BIASFACTOR=10
	TEMP=310
	LABEL=metad
	GRID_MIN=0,4.0
	GRID_MAX=10,10.0
	FILE=HILLS
... METAD

# output
PRINT ARG=cv_helix,cv_dist,metad.bias STRIDE=1000 FILE=COLVAR
"""

# --- add PLUMED force to system ---
system.addForce(PlumedForce(plumed_script))
simulation.context.reinitialize(preserveState=True)
simulation.reporters.clear()
simulation.reporters.append(DCDReporter('metad.dcd', 62500))
simulation.reporters.append(StateDataReporter('metad.log', 62500,
                                              step=True, speed=True))
# --- Metadynamics simulation --- 
total_steps = 50000000
checkpoint_freq = 2500000

for step in range(0, total_steps, checkpoint_freq):
    simulation.step(checkpoint_freq)
    simulation.saveCheckpoint('metad_restart.chk')
    print(f"Completed {int((step + checkpoint_freq)/250000)} ns of extension...")
