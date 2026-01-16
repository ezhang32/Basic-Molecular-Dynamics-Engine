import numpy as np
import matplotlib.pyplot as plt
from pymbar import timeseries
from vpython import canvas, sphere, vector, color, rate, box

class MDEngine:
    def __init__(self, n_particles=64, box_size=10.0, dt=0.002, gamma=1.0, kbt=1.0, sigma=1.0, visualize=True):
        self.N_PARTICLES = n_particles #number of particles
        self.BOX_SIZE = box_size #size of cubic box
        self.DT = dt #time step
        self.GAMMA = gamma #friction coefficient (higher = more slowdown, lower = more free movement) ~viscosity
        self.KBT = kbt #thermal energy
        self.SIGMA = sigma #particle diameter
        self.visualize = visualize #whether to show 3D visualization

        self.vel_history = []
        self.kinetic_history = []
        self.potential_history = []
        self.total_energy_history = []
        self.time_history = []
        self.t = 0.0

        self.rdf_data_list = []
        self.dr = 0.1

        # Initialize Canvas
        if self.visualize:
            self.scene = canvas(
                title='3D Molecular Dynamics',
                width=800,
                height=600,
                background=color.black,
                center=vector(self.BOX_SIZE / 2, self.BOX_SIZE / 2, self.BOX_SIZE / 2)
            )
            box(
                pos=vector(self.BOX_SIZE / 2, self.BOX_SIZE / 2, self.BOX_SIZE / 2),
                size=vector(self.BOX_SIZE, self.BOX_SIZE, self.BOX_SIZE),
                opacity=0.1,
                color=color.white
            )

        grid_dim = int(np.ceil(self.N_PARTICLES ** (1 / 3)))
        points = np.linspace(1, self.BOX_SIZE - 1, grid_dim)
        xv, yv, zv = np.meshgrid(points, points, points)
        self.pos = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T[:self.N_PARTICLES]
        self.vel = np.random.randn(self.N_PARTICLES, 3) * 0.5

        if visualize:
            self.atoms = [
            sphere(pos=vector(*self.pos[i]), radius=0.3, color=color.cyan)
            for i in range(self.N_PARTICLES)]
        else:
            self.atoms = []

    def get_distances(self):
        # Minimum Image Convention
        diff = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]
        diff -= self.BOX_SIZE * np.round(diff / self.BOX_SIZE)
        r_sq = np.sum(diff**2, axis=-1)
    
        # Use a mask to avoid division by zero and apply cutoff (3.0^2 = 9.0)
        mask = (r_sq > 0) & (r_sq < 9.0)
    
        # Calculate (sigma^2 / r^2)^3 which is (sigma/r)^6
        inv_r6 = np.zeros_like(r_sq)
        inv_r6[mask] = (self.SIGMA**2 / r_sq[mask])**3

        return diff, r_sq, inv_r6, mask

    def compute_forces(self):
        diff, r_sq, inv_r6, mask = self.get_distances()
    
        numerator = 48 * inv_r6 * (inv_r6 - 0.5)
        force_mag = np.divide(numerator, r_sq, out=np.zeros_like(numerator), where=r_sq != 0)

        return np.sum(diff * force_mag[..., np.newaxis], axis=1)
    
    def compute_potential_energy(self):
        _, _, inv_r6, mask = self.get_distances()
    
        # Simplified LJ Potential: 4 * (inv_r12 - inv_r6)
        potential = 4 * (inv_r6**2 - inv_r6)
        return 0.5 * np.sum(potential) # 0.5 to avoid double counting pairs

    def run_simulation(self, total_time):
        print("Simulation Running...")
        step = 0
        forces = self.compute_forces()
        
        while self.t < total_time:
            rate(60) #controls 3D visualization fps
            
            # VELOCITY VERLET INTEGRATION + LANGEVIN THERMOSTAT
            # 1. Update velocities by half-step
            sig_noise = np.sqrt(2.0 * self.KBT * self.GAMMA / self.DT)
            random_kick = np.random.normal(0, sig_noise, self.pos.shape)
            # (Langevin-style half step)
            self.vel += 0.5 * (forces + random_kick - self.GAMMA * self.vel) * self.DT
            # 2. Update positions by full step
            self.pos = (self.pos + self.vel * self.DT) % self.BOX_SIZE
            # 3. Recompute forces at new positions
            forces = self.compute_forces()
            # 4. Update velocities by second half-step
            self.vel += 0.5 * (forces + random_kick - self.GAMMA * self.vel) * self.DT
            
            # Update VPython visuals every few steps to save CPU
            if self.visualize and step % 5 == 0:
                for i in range(self.N_PARTICLES):
                    self.atoms[i].pos = vector(*self.pos[i])
                    speed = np.linalg.norm(self.vel[i])
                    # Mapping speed to RGB (roughly)
                    r = min(speed / 3.0, 1.0)
                    b = max(1.0 - speed / 3.0, 0.2)
                    self.atoms[i].color = vector(r, 0.4, b)

            # Data Collection
            ke = 0.5 * np.sum(self.vel ** 2)
            pe = self.compute_potential_energy()

            self.kinetic_history.append(ke)
            self.potential_history.append(pe)
            self.total_energy_history.append(ke + pe)
            self.time_history.append(self.t)
            self.vel_history.append(np.linalg.norm(self.vel, axis=1).copy())

            if step % 100 == 0: # Compute RDF every 100 steps to save time
                _, rdf = self.calculate_rdf()
                self.rdf_data_list.append((rdf, self.t))

            self.t += self.DT
            step += 1

    def calculate_rdf(self, dr=0.1):
        r_max = self.BOX_SIZE / 2.0
        n_bins = int(r_max / dr)

        diff = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]
        diff -= self.BOX_SIZE * np.round(diff / self.BOX_SIZE)
        r = np.linalg.norm(diff, axis=2)

        indices = np.triu_indices(self.N_PARTICLES, k=1)
        counts, bin_edges = np.histogram(r[indices], bins=n_bins, range=(0, r_max))

        r_inner, r_outer = bin_edges[:-1], bin_edges[1:]
        shell_volumes = (4 / 3) * np.pi * (r_outer ** 3 - r_inner ** 3)
        density = self.N_PARTICLES / (self.BOX_SIZE ** 3)

        rdf = (counts * 2) / (density * shell_volumes * self.N_PARTICLES)
        return r_inner + (dr / 2.0), rdf

    def calculate_coordination_number(self, rdf, dr=0.1):
        r = (np.arange(len(rdf)) + 0.5) * dr
        rho = self.N_PARTICLES / (self.BOX_SIZE ** 3)
        return np.cumsum(4 * np.pi * (r ** 2) * rdf * rho * dr) # Running coordination number

    def find_equilibrium_point(self):
        if len(self.potential_history) < 100:
            return 0
        t0, g, Neff_max = timeseries.detect_equilibration(np.array(self.potential_history))
        return t0

    def graphs(self, T_eff):
        t0 = self.find_equilibrium_point()
        t0_time = self.time_history[t0]

        fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

        # 1. Energy History
        axs[0, 0].plot(self.time_history, self.potential_history, label="Potential Energy")
        axs[0, 0].plot(self.time_history, self.total_energy_history, label="Total Energy", color='black')
        axs[0, 0].plot(self.time_history, self.kinetic_history, label="Kinetic Energy")
        axs[0, 0].axvline(x=t0_time, color='r', linestyle='--', label='Equilibration Point')
        axs[0, 0].set_title("System Energy vs. Time")
        axs[0, 0].set_xlabel(r"Time (reduced units $\tau$)")
        axs[0, 0].set_ylabel(r"Energy ($\epsilon$)")
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend(loc='upper right', fontsize='small', framealpha=0.5)

        # 2. Maxwell-Boltzmann
        all_speeds = np.concatenate(self.vel_history[t0:])
        v_range = np.linspace(0, all_speeds.max(), 100)
        mb = lambda v, T: (1 / (2 * np.pi * T)) ** 1.5 * 4 * np.pi * v ** 2 * np.exp(-v ** 2 / (2 * T))
        axs[0, 1].hist(all_speeds, bins=50, density=True, alpha=0.3, color='gray', label='Simulated Speeds')
        axs[0, 1].plot(v_range, mb(v_range, T_eff), 'r-', label=f'Measured (T={T_eff:.2f})')
        axs[0, 1].plot(v_range, mb(v_range, self.KBT), 'g--', label=f'Target (T={self.KBT})')
        axs[0, 1].set_title("Velocity Distribution")
        axs[0, 1].set_xlabel("Speed ($v$)")
        axs[0, 1].set_ylabel("Probability Density")
        axs[0, 1].legend()

        # 3. RDF & Coordination Number
        equil_rdfs = [rdf for rdf, time in self.rdf_data_list if time >= t0_time]
        if equil_rdfs:
            avg_rdf = np.mean(equil_rdfs, axis=0)
            r_vals = (np.arange(len(avg_rdf)) + 0.5) * self.dr
            cn_running = self.calculate_coordination_number(avg_rdf)

            ax_rdf = axs[1, 0]
            ax_cn = ax_rdf.twinx()
            l1, = ax_rdf.plot(r_vals, avg_rdf, color='purple', lw=2, label='g(r)')
            l2, = ax_cn.plot(r_vals, cn_running, color='green', linestyle='--', label='n(r)')
            
            ax_rdf.set_title("RDF and Running Coordination Number")
            ax_rdf.set_xlabel("Distance ($r/\sigma$)")
            ax_rdf.set_ylabel("Radial Distribution $g(r)$")
            ax_cn.set_ylabel("Coordination Number $n(r)$")
            # Combine legends from both axes
            ax_rdf.legend([l1, l2], ['g(r)', 'n(r)'], loc='upper center')

        # 4. Potential Energy Distribution
        axs[1, 1].hist(self.potential_history[t0:], bins=30, color='orange', alpha=0.7, edgecolor='black')
        axs[1, 1].set_title("Potential Energy Distribution (Equilibrated)")
        axs[1, 1].set_xlabel("Potential Energy ($U$)")
        axs[1, 1].set_ylabel("Frequency")
        axs[1, 1].grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25, top=0.95)
        plt.show()

    def analyze_temperature(self):
        t0 = self.find_equilibrium_point()
        avg_K = np.mean(self.kinetic_history[t0:])
        T_eff = (2.0 / (3.0 * self.N_PARTICLES)) * avg_K
        return T_eff

    def full_analysis(self):
        T_eff = self.analyze_temperature()
        self.graphs(T_eff)

    def get_data(self):
        return (
            self.vel_history,
            self.kinetic_history,
            self.potential_history,
            self.total_energy_history,
            self.time_history,
        )