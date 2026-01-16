# Basic-Molecular-Dynamics-Engine
MDEngine is a compact, extensible molecular dynamics (MD) engine designed for rapid experimentation, educational use, and scientific prototyping. It implements a 3D particle simulation with Langevin dynamics, customizable physical parameters, optional real‑time visualization, and automated post‑simulation analysis. This engine is ideal for users who want a transparent, easy‑to‑modify MD codebase without the overhead of large simulation frameworks.

**Key Features**
- Langevin Dynamics Integrator
Simulates particle motion under friction and stochastic thermal noise, enabling canonical ensemble sampling.
- Configurable Physical Parameters
Control particle count, box size, timestep, friction coefficient, thermal energy (KbT), and particle size.
- 3D Simulation Box
Particles evolve inside a cubic domain with periodic or reflective boundary handling (depending on your implementation).
- Optional VPython Visualization
Real‑time 3D rendering of particle trajectories for intuitive understanding of system behavior.
- Built‑in Data Logging
Tracks velocities, kinetic energy, potential energy, total energy, and time history throughout the simulation.
- Automated Analysis Tools
Includes velocity distribution histograms, Maxwell–Boltzmann fitting, radial distribution calculations, running coordinate numbers, and energy‑vs‑time diagnostics.

**Installation**
1. Place md_engine.py in your project directory
2. Run "pip install numpy matplotlib vpython" in bash to ensure necessary packages are installed.

**Methods**
This section documents every method in the MDEngine class, explaining its purpose, inputs, outputs, and role in the simulation workflow.

___init__(...)_ — Engine Initialization
Initializes the simulation environment and allocates all required data structures.
Responsibilities
- Stores user-defined parameters:
- n_particles, box_size, dt, gamma, kbt, sigma, visualize
- Initializes particle positions and velocities
- Allocates arrays for forces, energy tracking, and speed logging
- Sets up visualization objects if enabled

_get_distances()_ — Pairwise Distance Matrix
Computes the Euclidean distances between all particle pairs.
Responsibilities
- Returns an (N, N) matrix of scalar distances
- Used internally for force and energy calculations

_compute_forces()_ — Interparticle Force Calculation
Calculates net forces on each particle based on pairwise interactions.
Responsibilities
- Uses distance matrix to evaluate repulsive or attractive forces
- Returns a (N, 3) array of force vectors

_compute_potential_energy()_ — Total Potential Energy
Calculates the system's total potential energy from pairwise interactions.
Responsibilities
- Uses distance matrix and interaction potential
- Returns a scalar energy value

_run_simulation(total_time)_ — Main Simulation Loop
Executes the MD integration for the specified duration.
Responsibilities
- Advances the system using Langevin dynamics
- Logs energy and velocity data
- Updates visualization if enabled

_calculate_rdf()_ — Radial Distribution Function
Analyzes spatial structure by computing the radial distribution function g(r).
Responsibilities
- Bins pairwise distances into a histogram
- Normalizes by ideal gas reference
- Returns RDF curve for structural analysis

_calculate_coordination_number()_ — Local Environment Analysis
Computes the average number of neighbors within a cutoff radius.
Responsibilities
- Uses distance matrix and cutoff threshold
- Returns coordination number per particle and system average

_find_equilibrium_point()_ — Equilibration Detection
Identifies when the system reaches thermal or energetic equilibrium.
Responsibilities
- Analyzes energy or temperature time series
- Returns estimated timestep or time of equilibration

_graphs()_ — Plotting Utilities
Generates visualizations of simulation diagnostics.
Responsibilities
- Plots energy vs. time, RDF, coordination number, etc.
- Saves or displays figures

_analyze_temperature()_ — Thermodynamic Diagnostics
Computes effective temperature from kinetic energy.
Responsibilities
- Finds T = 2(Ke)/[3NKb]
- Compares to target kbt
- Returns temperature time series and average

_full_analysis()_ — Post-Simulation Summary
Runs all major diagnostics and visualizations.
Responsibilities
- Calls:
- analyze_temperature()
- calculate_rdf()
- calculate_coordination_number()
- graphs()
- Prints summary statistics and saves plots

_get_data()_ — Retrieve Raw Simulation Data
Returns stored arrays for further analysis or export.
Responsibilities
- Provides access to:
- Particle positions
- Velocities
- Energies
- Speed distributions

**Special Notes**
Simulations can take anywhere from 30 seconds to 10 minutes, depending on the time steps, simulation time, and temperature. It's recommended that for KbT <= 1, run simulations with dt = 0.002 and simulation_time >= 25.0. Not doing so may cause the system not to equilibrate and/or the physics breaks down. 


