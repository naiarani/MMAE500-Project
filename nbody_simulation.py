import numpy as np
from scipy.integrate import solve_ivp
import os

# simulation givens
G = 1.0 #gravity
L = 3.0 #LxL box
dt, Nt = 0.01, 1000
t_sim = np.linspace(0, dt*(Nt-1), Nt)

def n_body_rhs(t, state, n, masses):
    pos = state[:2*n].reshape(n,2)
    vel = state[2*n:4*n].reshape(n,2)
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            rij = pos[i]-pos[j]
            acc[i] += -G*masses[j]*rij/np.linalg.norm(rij)**3
    return np.hstack((vel.flatten(), acc.flatten()))

def random_initial_conditions(L, v_scale, n, masses):
    pos0 = np.random.uniform(-L,L,(n,2))
    vel0 = np.random.uniform(-v_scale,v_scale,(n,2))
    Mtot = masses.sum()
    COM_r = (masses[:,None]*pos0).sum(0)/Mtot
    COM_v = (masses[:,None]*vel0).sum(0)/Mtot
    pos0 -= COM_r; vel0 -= COM_v
    return np.hstack((pos0.flatten(), vel0.flatten()))

# Loop over number of bodies 
for n in range(3, 4):
    masses = np.ones(n)  # masses=1
    Mtot = masses.sum()  # equals n when masses are 1
    v_scale = np.sqrt(G * Mtot / L)

    num_simulations = 10000  # change this!!!!
    save_dir = f'simulations/nbody_{n:02d}_10000'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n--- Running {num_simulations} simulations for {n}-body system ---")

    for sim_idx in range(num_simulations):
        x0 = random_initial_conditions(L, v_scale, n, masses)

        sol = solve_ivp(
            n_body_rhs, (0, t_sim[-1]), x0, t_eval=t_sim,
            args=(n, masses), rtol=1e-9, atol=1e-9
        )
        data_sim = sol.y.T  # (Nt,4n)

        # Save trajectory 
        np.savez(f'{save_dir}/sim_{sim_idx:03d}.npz',
                 t_sim=t_sim, data_sim=data_sim,
                 initial_conditions=x0, masses=masses)

        if (sim_idx+1) % 50 == 0 or sim_idx == num_simulations-1:
            print(f"Completed {sim_idx+1}/{num_simulations} simulations for {n}-body")

print("\n✅ All simulations completed and saved.")

