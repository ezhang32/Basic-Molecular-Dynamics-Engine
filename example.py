import time
from md_engine import MDEngine

start_time = time.time()

# Initialize Simulation Engine with parameters
md_engine = MDEngine(
    n_particles= 100,
    box_size=10.0,
    dt=0.01,
    gamma=1.0,
    kbt=2,
    sigma=1.0,
    visualize=False
)

# Run for 20 time units (2000 time steps with dt=0.01)
md_engine.run_simulation(total_time=20.0)

# Measure and print execution time
end_time = time.time()
duration = end_time - start_time
print(f"Simulation completed in: {duration:.2f} seconds")
print("Retrieving and analyzing data...")

# Plotting and analysis
md_engine.full_analysis()
