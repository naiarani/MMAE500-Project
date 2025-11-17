import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as ag
import matplotlib.pyplot as plt
import glob, os


# Select n-body simulation to train on
n = 2
sim_dir = f'simulations/nbody_{n:02d}'
sim_file = glob.glob(os.path.join(sim_dir, '*.npz'))[0]  # pick first sim for example

data = np.load(sim_file)
t_sim = data['t_sim']
data_sim = data['data_sim']

t_data = torch.tensor(t_sim, dtype=torch.float32).unsqueeze(1)
x_data = torch.tensor(data_sim, dtype=torch.float32)

G = 1.0
masses = np.ones(n)

# PINN Architecture (MLP)
class PINN(nn.Module):
    def __init__(self, n, hidden=[128,128]):
        super().__init__()
        dims = [1] + hidden + [4*n]
        layers = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i],dims[i+1]), nn.Tanh()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self,t):
        return self.net(t)

model = PINN(n).float()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, min_lr=1e-6)
mse = nn.MSELoss()

Ncol = 1000 # collocation points
t_col = torch.tensor(np.random.uniform(0, t_sim[-1], (Ncol,1)),
                     requires_grad=True, dtype=torch.float32)

# Loss Functions: Data, initial conditions, physics. Also includes the loss stages here...
def data_loss():
    return mse(model(t_data), x_data)

def ic_loss():
    x0_pred = model(torch.zeros(1,1))
    return mse(x0_pred, x_data[0:1])

def physics_loss():
    x_pred = model(t_col)
    x_t = torch.cat([
        ag.grad(x_pred[:,j], t_col, grad_outputs=torch.ones_like(x_pred[:,j]), create_graph=True)[0].unsqueeze(1)
        for j in range(4*n)], dim=1)
    x_tt = torch.cat([
        ag.grad(x_t[:,j], t_col, grad_outputs=torch.ones_like(x_t[:,j]), create_graph=True)[0].unsqueeze(1)
        for j in range(4*n)], dim=1)

    pos_pred = x_pred[:, :2*n].view(-1,n,2)
    acc_pred = x_tt[:, :2*n].view(-1,n,2)

    loss_p = 0.0
    for i in range(n):
        force_sum = torch.zeros_like(acc_pred[:,i])
        for j in range(n):
            if i==j: continue
            rij = pos_pred[:,i]-pos_pred[:,j]
            force_sum += -G*masses[j]*rij/(rij.norm(dim=1,keepdim=True)**3)
        loss_p += mse(acc_pred[:,i], force_sum)
    return loss_p

# Multi-phase loss Training
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()

    if epoch < 1000:    # Phase 1: Data & IC only
        loss = data_loss() + 1000*ic_loss()
    elif epoch < 4000:  # Phase 2: Ramp up physics
        frac = (epoch-1000)/3000
        loss = data_loss() + (frac)*physics_loss() + 1000*ic_loss()
    else:               # Phase 3: All losses
        loss = data_loss() + physics_loss() + 1000*ic_loss()

    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.3e}")

# evaluation
model.eval()
with torch.no_grad():
    x_pinn = model(t_data).cpu().numpy()

# RMSE calculation
rmse = np.sqrt(np.mean((x_pinn - data_sim)**2))
print(f'Overall RMSE: {rmse:.3e}')

# plotting for trajectories
plt.figure(figsize=(6,6))
for i in range(n):
    plt.plot(data_sim[:,2*i], data_sim[:,2*i+1], '-', label=f'True Body {i+1}')
    plt.plot(x_pinn[:,2*i], x_pinn[:,2*i+1], '--', label=f'PINN Body {i+1}')
plt.legend()
plt.xlabel('X'); plt.ylabel('Y')
plt.title('Trajectories: True vs PINN')
plt.axis('equal')
plt.show()

# Compute error for each body
Nt = data_sim.shape[0]
n = data_sim.shape[1] // 4  # each body has x, y, vx, vy
errors = np.zeros((Nt, n))

for i in range(n):
    true_xy = data_sim[:, 2*i:2*i+2]   # true x, y
    pred_xy = x_pinn[:,  2*i:2*i+2]    # predicted x, y
    errors[:, i] = np.linalg.norm(pred_xy - true_xy, axis=1)

# Plot error over time
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
for i in range(n):
    plt.plot(t_sim, errors[:, i], label=f'Body {i+1} error')
plt.xlabel('t')
plt.ylabel('‖PINN − True‖₂ (m)')
plt.title('Per-Body Trajectory Error Over Time')
plt.legend()
plt.grid(True)
plt.show()