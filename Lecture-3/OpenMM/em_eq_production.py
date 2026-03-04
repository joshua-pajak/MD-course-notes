# --- Import libraries ---
import os
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

# --- Input files ---
inpcrd = AmberInpcrdFile('input.rst7')
prmtop = AmberPrmtopFile('topol.prmtop', periodicBoxVectors=inpcrd.boxVectors)

# --- System configuration ---
nonbondedMethod = PME
nonbondedCutoff = 1.0*nanometers
ewaldErrorTolerance = 0.0005
constraints = HBonds
rigidWater = True
constraintTolerance = 0.000001

# --- Integration options ----
dt = 0.002*picoseconds
temperature = 310.15*kelvin
friction = 1.0/picosecond
pressure = 1.0*atmospheres
barostatInterval = 25

# --- Simulation options ---
steps = 100000
equilibrationSteps = 100000
platform = Platform.getPlatformByName('CUDA')
platformProperties = {'Precision': 'mixed'}
equilibrationDcdReporter = DCDReporter('eq.dcd', 1000)
productionDcdReporter = DCDReporter('prod.dcd', 1000)
productionDataReporter = StateDataReporter('log.txt', 1000, totalSetps=steps,
                                           step=True, speed=True, progress=True,
                                           potentialEnergy=True, temperature=True, separator='\t')
ProductionCheckpointReporter = CheckpointReporter('checkpoint.chk', 1000)

# --- Prepare the simulation ---
print('Building system...')
system = prmtop.createSystem(nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
                             constraints=constraints, rigidWater=rigidWater, 
                             ewaldErrorTolerance=ewaldErrorTolerance)
integrator = LangevinMiddleIntegrator(temperature, friction, dt)
integrator.setConstraintTolerance(constraintTolerance)
simulation = Simulation(prmtop.topology, system, integrator, platform, platformProperties)
simulation.context.setPositions(inpcrd.positions)

# --- Minimization and equilibration ---
print('Performing energy minimization...')
simulation.minimizeEnergy()
print('Equilibrating...')
simulation.context.setVelocitiesToTemperature(temperature)
simulation.reporters.append(equilibrationDcdReporter)
simulation.step(equilibrationSteps)

# --- Production simulation ---
print('Simulating...')
simulation.reporters.clear()
system.addForce(MonteCarloBarostat(pressure, temperature, barostatInterval))
system.context.reinitialize(preserveState=True)
simulation.reporters.append(productionDcdReporter)
simulation.reporters.append(productionDataReporter)
simulation.reporters.append(checkpointreporter)
simulation.currentStep = 0
simulation.step(steps)