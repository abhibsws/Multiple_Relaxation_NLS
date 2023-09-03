#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Diff_Time_Integrators import Compute_Sol_ImEx, Compute_Sol_ImEx_Relaxation, sol_nl_part, sol_l_part, Op_Split_Exact_Solve, ImEx_schemes, Op_Sp_Coeff



def nonlin_f(z,ep,case):
    if case == 'weak_cubic_defoc':
        y = ep*z
    elif case == 'weak_cubic_foc':
        y = -ep*z 
    elif case == 'strong_cubic_foc_zero_phase':
        y = -z 
    elif case == 'strong_cubic_foc_nonzero_phase':
        y = -z 
    return y

def A0_fun(x,case):
    if case == 'weak_cubic_defoc':
        y = np.exp(-x**2)
    elif case == 'weak_cubic_foc':
        y = np.exp(-x**2)
    elif case == 'strong_cubic_foc_zero_phase':
        y = np.exp(-x**2)
    elif case == 'strong_cubic_foc_nonzero_phase':
        y = np.exp(-x**2)
    return y

def S0_fun(x,ep,case):
    if case == 'weak_cubic_defoc':
        y = -x**2/2 + ep*np.exp(-2*x**2)*np.log(1/ep)
    elif case == 'weak_cubic_foc':
        y =  -np.log(np.exp(x)+np.exp(-x))
    elif case == 'strong_cubic_foc_zero_phase':
        y = x*0
    elif case == 'strong_cubic_foc_nonzero_phase':
        y = 1/(np.exp(x)+np.exp(-x))
    return y

def initial_cond(x,ep,case):
    ucomp = A0_fun(x,case)*np.exp(1j*S0_fun(x,ep,case)/ep)
    return np.concatenate((ucomp.real,ucomp.imag)).astype('float64')

# Stiff part of the right hand side function of the ODE system in spatial doamin
def rhsNlsStiff(u,N,xi,ep):
    ucomp = u[:N]+ 1j *u[N:]
    uhat = np.fft.fft(ucomp)
    duhat = -1j*(ep/2)*xi**2*uhat
    rhs_comp = np.fft.ifft(duhat)
    dudt = np.concatenate((rhs_comp.real,rhs_comp.imag)).astype('float64')
    return dudt

# Non stiff part of right hand side function of the ODE system in spatial doamin
def rhsNlsNonStiff(u,N,ep,case):
    ucomp = u[:N]+ (1j) *u[N:]
    rhs_comp= -(1j/ep)*nonlin_f(np.abs(ucomp)**2,ep,case)*ucomp
    dudt = np.concatenate((rhs_comp.real,rhs_comp.imag)).astype('float64')
    return dudt

def position_density(u,N):  # n(x,t)
    V = u[:N]; W = u[N:]
    return V**2+W**2

def current_density(u,N):  # J(x,t)
    V = u[:N]; W = u[N:]
    return np.multiply(V[:-1],W[1:])-np.multiply(V[1:],W[:-1])

# For multi-relaxation
def rgam(gamma,u,N,dx,ep,case,inc,inv,E_old):
    uprop = u + np.dot(gamma,inc)  
    if inv == 1:
        E_prop = np.array([eta1(uprop,N,dx)])
    elif inv == 2:
        E_prop = np.array([eta1(uprop,N,dx),eta2(uprop,N,dx,ep,case)])
    return E_prop-E_old

def eta1(u,N,dx):  # Invariant: first quantity
    V = u[:N]; W = u[N:]
    return dx*np.sum(V**2+W**2)


def eta2(u,N,dx,ep,case):  # Current density
    V = u[:N]; W = u[N:]
    return dx*np.sum(np.multiply(V[:-1],W[1:])-np.multiply(V[1:],W[:-1]))



# Different cases
case = 'strong_cubic_foc_nonzero_phase'; case_title = 'Strong Cubic Foc Nonzero Phase';
xL = -8; xR =8; L = xR-xL; N_ref = L*4096; dt_ref = 1e-4; t0 = 0; 
Eps = np.array([0.2,0.2/2,0.2/4]); Ts = np.array([0.5,0.9])
# Operator splitting methods
OS_method_names = ['Op_Sp1','Op_Sp2','Op_Sp4']; OS_Stage = [1,2,5]; OS_Order = [1,2,4]; OS_Sch_No = [0,1,2];



St_foc_Nz_Phase = np.empty((len(Eps), len(Ts), 2*N_ref)); OS_idx = 2;
OpSp4_a, OpSp4_b = Op_Sp_Coeff(OS_Stage[OS_idx],OS_Order[OS_idx],OS_Sch_No[OS_idx])
for i in range(len(Eps)):
    for j in range(len(Ts)):
        f_stiff = rhsNlsStiff; f_non_stiff = rhsNlsNonStiff; IC = initial_cond
        sp_ref_tt_4, sp_ref_uu_4 = Op_Split_Exact_Solve(nonlin_f,sol_nl_part,sol_l_part,IC,Eps[i],case,xL,xR,                                             N_ref,t0,Ts[j],OS_method_names[OS_idx],OpSp4_a,OpSp4_b,dt_ref)
        St_foc_Nz_Phase[i,j] = sp_ref_uu_4[-1]
        

from pathlib import Path 
data = {'Method': OS_method_names[OS_idx],
        'Domain':'[%d,%d]'%(xL,xR),
        'N_ref': N_ref,
        'dt_ref': dt_ref}

df = pd.DataFrame(data, index=[0])

# saving data and information
filepath = Path("./%s_RefSol_N_%d_dt_%1.e_Data.csv"%(case,N_ref,dt_ref),index = False)    
df.to_csv(filepath)         
np.save("./%s_RefSol_N_%d_dt_%1.e.npy"%(case,N_ref,dt_ref), St_foc_Nz_Phase)

