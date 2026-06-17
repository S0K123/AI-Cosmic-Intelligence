import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from data_loader import load_data
from feature_engineering import clean_and_prepare_data
from train_model import create_habitability_label

# Gravitational Constant in AU^3 / (Solar Mass * Day^2)
G = 0.0002959122082855911

class CelestialBody:
    def __init__(self, name, mass, pos, vel, color='blue', size=10):
        self.name = name
        self.mass = mass  # in Solar Masses
        self.pos = np.array(pos, dtype=float)  # in AU [x, y]
        self.vel = np.array(vel, dtype=float)  # in AU/day [vx, vy]
        self.color = color
        self.size = size
        self.path_x = []
        self.path_y = []

    def update_path(self):
        self.path_x.append(self.pos[0])
        self.path_y.append(self.pos[1])
        # Keep path length manageable
        if len(self.path_x) > 500:
            self.path_x.pop(0)
            self.path_y.pop(0)

def get_gravitational_acceleration(body, other_bodies):
    acc = np.zeros(2)
    for other in other_bodies:
        if body == other:
            continue
        r_vec = other.pos - body.pos
        r_mag = np.linalg.norm(r_vec)
        if r_mag == 0:
            continue
        # Newton's Law of Gravitation: F = G * m1 * m2 / r^2
        # a = F / m1 = G * m2 / r^2
        acc += G * other.mass * r_vec / (r_mag**3)
    return acc

def simulate_step(bodies, dt):
    """
    Update positions and velocities using Velocity Verlet integration.
    1. pos(t + dt) = pos(t) + vel(t)*dt + 0.5*acc(t)*dt^2
    2. acc(t + dt) = f(pos(t + dt))
    3. vel(t + dt) = vel(t) + 0.5*(acc(t) + acc(t + dt))*dt
    """
    # Store initial accelerations
    acc_initial = {b: get_gravitational_acceleration(b, bodies) for b in bodies}
    
    # 1. Update positions
    for b in bodies:
        b.pos += b.vel * dt + 0.5 * acc_initial[b] * (dt**2)
        b.update_path()
        
    # 2. Update velocities
    for b in bodies:
        acc_final = get_gravitational_acceleration(b, bodies)
        b.vel += 0.5 * (acc_initial[b] + acc_final) * dt

def setup_simulation_system(system_name='TRAPPIST-1'):
    """
    Load data for a specific system and initialize CelestialBody objects.
    """
    raw_df = load_data()
    if raw_df is not None:
        if 'default_flag' in raw_df.columns:
            raw_df = raw_df[raw_df['default_flag'] == 1].copy()
        
        # We need unscaled data for physics
        df = clean_and_prepare_data(raw_df)
        df = create_habitability_label(df)
        
        system_df = df[df['planet_name'].str.contains(system_name, na=False)]
        if system_df.empty:
            print(f"System {system_name} not found, using a default mock system.")
            return None
        
        star_mass = system_df['st_mass'].iloc[0] if 'st_mass' in system_df.columns else 1.0
        # If mass is NaN, default to 1.0
        if np.isnan(star_mass): star_mass = 1.0
        
        bodies = []
        # Add Star at center
        bodies.append(CelestialBody(system_name, star_mass, [0, 0], [0, 0], color='yellow', size=100))
        
        # Add Planets
        for _, row in system_df.iterrows():
            r = row['orbit_semi_major_axis']
            if np.isnan(r): continue
            
            # Circular orbit velocity: v = sqrt(G * M / r)
            v_mag = np.sqrt(G * star_mass / r)
            
            # Start planet at [r, 0] with velocity [0, v_mag]
            color = 'cyan' if row['habitability_label'] == 1 else 'blue'
            bodies.append(CelestialBody(row['planet_name'], 0.000003, [r, 0], [0, v_mag], color=color, size=20))
            
        return bodies
    return None

def run_simulation(system_name='TRAPPIST-1'):
    bodies = setup_simulation_system(system_name)
    if not bodies:
        print("Simulation setup failed.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('black')
    
    # Find max distance to set limits
    max_r = max([np.linalg.norm(b.pos) for b in bodies]) * 1.2
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_aspect('equal')
    ax.set_title(f"Space Simulation: {system_name} System", color='white')

    # Visual elements
    scatters = [ax.scatter([], [], s=b.size, c=b.color, label=b.name) for b in bodies]
    lines = [ax.plot([], [], color=b.color, alpha=0.3, lw=1)[0] for b in bodies]
    
    # Legend
    leg = ax.legend(loc='upper right', fontsize='small', frameon=True)
    plt.setp(leg.get_texts(), color='white')

    dt = 0.5  # Simulation step size in days

    def animate(i):
        # Run multiple physics steps per frame for smoothness
        for _ in range(5):
            simulate_step(bodies, dt)
        
        for idx, b in enumerate(bodies):
            scatters[idx].set_offsets(b.pos)
            lines[idx].set_data(b.path_x, b.path_y)
        return scatters + lines

    print(f"Starting animation for {system_name}...")
    ani = FuncAnimation(fig, animate, frames=200, interval=30, blit=True)
    
    # Save as GIF (requires pillow)
    print("Saving simulation to plots/space_simulation.gif...")
    try:
        ani.save('plots/space_simulation.gif', writer='pillow')
        print("Simulation saved successfully.")
    except Exception as e:
        print(f"Could not save GIF: {e}")
        plt.show()

if __name__ == "__main__":
    print("Initializing Simulation...")
    run_simulation('TRAPPIST-1')
