#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 01:06:44 2020

@author: saul
"""
import torch
from Utils import rk4_odeint, wrap_to_pi
class double_pendulum(object):
        def __init__(self):
            self.g = 9.8
            self.m1 = 1 
            self.m2 = 1
            self.l1 = 1
            self.l2 = 1
        def f_analytical(self,state):
            t1, t2, w1, w2 = torch.split(state)
            a1 = (self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * torch.cos(t1 - t2)
            a2 = (self.l1 / self.l2) * torch.cos(t1 - t2)
            f1 = -(self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * (w2**2) * torch.sin(t1 - t2) - (self.g /self. l1) * torch.sin(t1)
            f2 = (self.l1 / self.l2) * (w1**2) * torch.sin(t1 - t2) - (self.g / self.l2) * torch.sin(t2)
            g1 = (f1 - a1 * f2) / (1 - a1 * a2)
            g2 = (f2 - a2 * f1) / (1 - a1 * a2)
            return torch.stack((w1, w2, g1, g2),0)
        
        
        
class generate_trajectory(object):
    def __init__(self, h, N, initial_condition, model, normalize=None):
        self.h = h
        self.initial_condition=initial_condition
        self.model=model
        self.N=N
        self.n_dim=initial_condition.size()
        self.normalize=normalize
    def get_trajectory(self):

        states=torch.empty(size=(self.N, self.n_dim))
        states[0]=self.initial_condition
        for i in range(self.N):
            if self.normalize:
                x_next=wrap_to_pi(rk4_odeint(self.model.f_analytical, states[i], self.h, self.n_dim))
            else:
                x_next=rk4_odeint(self.model, states[i], self.h, self.n_dim)
            states[i+1]=x_next
        return states
    
    
x0=torch.tensor([0,0,0,0])
model=double_pendulum()
x=generate_trajectory(0.01, 10, x0, model).get_trajectory()

            
