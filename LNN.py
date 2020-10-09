#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 06:12:55 2020

@author: saul
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:35:14 2020
Copied and adapted from https://torchdyn.readthedocs.io/en/latest/tutorials/09_lagrangian_nets.html
@author: saul
"""
import torch
from torch import nn
from torch.autograd import grad
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import matplotlib.pyplot as plt

class LNN(nn.Module):
    def __init__(self):
        super().__init__()
        h_dim = 500
        self.fc1 =l1= nn.Linear(4, h_dim)
        self.fc2 =l2= nn.Linear(h_dim,h_dim)

        self.fc3 =l3= nn.Linear(h_dim,h_dim)

        self.fc_last=l4 = nn.Linear(h_dim,1)

        
    def forward(self, x):
        with torch.set_grad_enabled(True):
            qqd = x.requires_grad_(True)
            time_step = torch.tensor(0.01)
            out=self._rk4_step(qqd,time_step)
        return out
   
    def _lagrangian(self, qqd):
        x = F.softplus(self.fc1(qqd))
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        L = self.fc_last(x)
        return L
    
    def euler_lagrange(self,qqd):
        self.n = n = qqd.shape[1]//2
        L = self._lagrangian(qqd).sum()
        J = grad(L, qqd, create_graph=True)[0] ;
        DL_q, DL_qd = J[:,:n], J[:,n:]
        DDL_qd = []
        for i in range(n):
            J_qd_i = DL_qd[:,i][:,None]
            H_i = grad(J_qd_i.sum(), qqd, create_graph=True)[0][:,:,None]
            DDL_qd.append(H_i)
        DDL_qd = torch.cat(DDL_qd, 2)
        DDL_qqd, DDL_qdqd = DDL_qd[:,:n,:], DDL_qd[:,n:,:]
        T = torch.einsum('ijk, ij -> ik', DDL_qqd, qqd[:,n:])
        qdd = torch.einsum('ijk, ij -> ik', DDL_qdqd.pinverse(), DL_q - T)
        return torch.cat([qqd[:,self.n:], qdd], 1)
    
    
    def _rk4_step(self, qqd, h=None):
        k1 = h * self.euler_lagrange(qqd)
        k2 = h * self.euler_lagrange(qqd + k1/2)
        k3 = h * self.euler_lagrange(qqd + k2/2)
        k4 = h *self.euler_lagrange(qqd + k3)
        return qqd + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
class Baseline_FF_Network(nn.Module):
    def __init__(self):
        super().__init__()
        h1_dim = 500
        h2_dim = 500
        self.fc1 = nn.Linear(4, h1_dim)
        self.fc2 = nn.Linear(h1_dim,h2_dim)
        self.fc3 = nn.Linear(h1_dim,h2_dim)
        self.fc_last = nn.Linear(h2_dim, 4)

    def forward(self,qqd):
        x = F.softplus(self.fc1(qqd))
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        x = self.fc_last(x)
        return x
    
def train(model, criterion, trainloader, device, optimizer, scheduler, num_epoch, testloader, step_lr=None):
    losses=[]
    lrs=[]
    for i in range(num_epoch):
        model.train()
        running_loss = []
        for state, target in trainloader:
            state = state.to(device)
            target = target.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(state)
            loss = criterion(pred, target) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step() # Update trainable weights
            if step_lr==False:
                scheduler.step() #Then OneCycleLr
        if step_lr==True:
            scheduler.step()
        lrs.append([g['lr'] for g in optimizer.param_groups])
        losses.append(np.mean(running_loss))
        Ave_MSE=evaluate(model, criterion, testloader, device)
        
        print("Epoch {}: Train_loss:{} Test_loss: {} Lr:{}".format(i+1,np.mean(running_loss), Ave_MSE, [g['lr'] for g in optimizer.param_groups])) # Print the average loss for this epoch
    return losses,lrs


def evaluate(model, criterion, loader, device): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    MSEs = []
    with torch.no_grad(): # Do not calculate gradient to speed up computation
        for state, target in (loader):
            state = state.to(device)
            target = target.to(device)
            pred = model(state)

            MSE_error = criterion(pred, target)
            MSEs.append(MSE_error.item())

    Ave_MSE = np.mean(np.array(MSEs))
    return Ave_MSE


def db_analytical(state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
    t1, t2, w1, w2 = state
    a1 = (l2 / l1) * (m2 / (m1 + m2)) * torch.cos(t1 - t2)
    a2 = (l1 / l2) * torch.cos(t1 - t2)
    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * torch.sin(t1 - t2) - (g / l1) * torch.sin(t1)
    f2 = (l1 / l2) * (w1**2) * torch.sin(t1 - t2) - (g / l2) * torch.sin(t2)
    g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1 - a1 * a2)
    return torch.stack((w1, w2, g1, g2),0)

def rk4_step(f, x, t, h):
    # one step of Runge-Kutta integration
    k1 = h * f(x, t)
    k2 = h * f(x + k1/2, t + h/2)
    k3 = h * f(x + k2/2, t + h/2)
    k4 = h * f(x + k3, t + h)
    return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

def normalize(state):
    # wrap generalized coordinates to [-pi, pi]
    pi = torch.acos(torch.zeros(1)).item() * 2
    return torch.cat(((state[:2] + pi) % (2 * pi) - pi, state[2:]),0)

N_train=200 #number of training points
N_test=1000 #number of test points

time_step=0.01 #time step for the rk4 discretization
num_epochs =100 # Choose an appropriate number of training epochs
batch_size=200 #Define batch size. The typically mini-batch sizes are 64, 128, 256 or 512. If it is too large may lead Cuda out of memory
weight_decay=0 #weight_decay is L2 regularization strength


device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
torch.manual_seed(0)

LNN_model = LNN().to(device)
criterion = nn.MSELoss() #Loss Function. Default:mean

'''
Test Data
'''
torch.manual_seed(1)
time_step = 0.01
q=torch.distributions.Uniform(-3.14, +3.14).sample((N_test, 2))
qd=torch.distributions.Uniform(-10,10).sample((N_test, 2))
xtest=torch.cat((q,qd),1)
n_input=xtest.shape[1]
ytest= torch.empty(size=(N_test, n_input))

for i in range(N_test):
    xnext=rk4_step(db_analytical, xtest[i], 0, time_step)
    ytest[i]=xnext

test_data = TensorDataset(xtest, ytest)
testloader = DataLoader(test_data, batch_size=N_test, shuffle=False)
    
'''
Train Data
'''        
torch.manual_seed(2)
q=torch.distributions.Uniform(-3.14,3.14).sample((N_train, 2))
qd=torch.distributions.Uniform(-10, 10).sample((N_train, 2))
xtrain=torch.cat((q,qd),1)
n_input=xtrain.shape[1]
ytrain= torch.empty(size=(N_train, n_input))
for i in range(N_train):
    xnext=rk4_step(db_analytical, xtrain[i], 0, time_step)
    ytrain[i]=xnext

train_data = TensorDataset(xtrain, ytrain)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

#Step Learning Rate:
lr=1e-2 #Define initial learning rate. Typically there is a better convergence for initial_lr=1e-2 and 1e-3.
step_size=10 #Define stepsize for learning rate change
gamma=0.5 #Multiplicative factor ---> less than 1

optimizer = optim.Adam(LNN_model.parameters(), lr=lr, weight_decay=weight_decay) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

losses,lr=train(LNN_model, criterion, trainloader, device, optimizer, scheduler, num_epochs, testloader, True)

#Make Plots
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.plot(lr,losses)
plt.ylabel('Loss')
plt.xlabel('Lr')
plt.show()
plt.plot(lr)
plt.ylabel('Lr')
plt.xlabel('Epoch')
plt.show()

#One Cycle Learning Rate:
max_lr=1e-2
steps_per_epoch=1
final_div_factor =10
div_factor=10

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=num_epochs, final_div_factor =final_div_factor, div_factor=div_factor)
losses,lr=train(LNN_model, criterion, trainloader, device, optimizer, scheduler, num_epochs, testloader, False)

plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.plot(lr,losses)
plt.ylabel('Loss')
plt.xlabel('Lr')
plt.show()
plt.plot(lr)
plt.ylabel('Lr')
plt.xlabel('Epoch')
plt.show()

BB_model = Baseline_FF_Network().to(device)


#Step Learning Rate:
lr=1e-3 #Define initial learning rate. Typically there is a better convergence for initial_lr=1e-2 and 1e-3.
step_size=100 #Define stepsize for learning rate change
gamma=0.9 #Multiplicative factor ---> less than 1

optimizer = optim.Adam(BB_model.parameters(), lr=lr, weight_decay=weight_decay) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

losses,lr=train(BB_model, criterion, trainloader, device, optimizer, scheduler, 1000, testloader, True)

#Make Plots
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.plot(lr,losses)
plt.ylabel('Loss')
plt.xlabel('Lr')
plt.show()
plt.plot(lr)
plt.ylabel('Lr')
plt.xlabel('Epoch')
plt.show()

# %%

def kinetic_energy(state, m1=1, m2=1, l1=1, l2=1, g=9.8):
    q, q_dot = torch.split(state, 2)
    (t1, t2), (w1, w2) = q, q_dot

    T1 = 0.5 * m1 * (l1 * w1)**2
    T2 = 0.5 * m2 * ((l1 * w1)**2 + (l2 * w2)**2 + 2 * l1 * l2 * w1 * w2 * torch.cos(t1 - t2))
    T = T1 + T2
    return T

def potential_energy(state, m1=1, m2=1, l1=1, l2=1, g=9.8):
    q, q_dot = torch.split(state, 2)
    (t1, t2), (w1, w2) = q, q_dot

    y1 = -l1 * torch.cos(t1)
    y2 = y1 - l2 * torch.cos(t2)
    V = m1 * g * y1 + m2 * g * y2
    return V


x0=torch.tensor([[np.pi/2, np.pi/4,0,0]])
N=2000
n_input=xtest.shape[1]
x_traj= torch.empty(size=(N, n_input))
x_traj[0]=x0
for i in range(N):
    x_next=LNN_model(x_traj[i])
    x_traj[i+1]=xnext

