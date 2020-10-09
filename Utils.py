#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 00:54:02 2020

@author: saul
"""

import torch


def rk4_odeint(f, x, h):
    # one step of Runge-Kutta integration
    k1 = h * f(x)
    k2 = h * f(x + k1/2)
    k3 = h * f(x + k2/2)
    k4 = h * f(x + k3)
    return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)


def wrap_to_pi(state, n_dim):
    # wrap generalized coordinates to [-pi, pi]
    pi = torch.acos(torch.zeros(1)).item() * 2
    return torch.cat(((state[:n_dim//2] + pi) % (2 * pi) - pi, state[n_dim//2:]),0)