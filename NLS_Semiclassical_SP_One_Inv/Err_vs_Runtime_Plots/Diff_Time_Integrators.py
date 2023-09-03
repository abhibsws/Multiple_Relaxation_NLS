#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.optimize import fsolve,brute,fmin
from IPython.display import clear_output, display, Math
from time import time

# This is an implementation of an ImEx time integrator
def Compute_Sol_ImEx(f_stiff,f_non_stiff,IC,ep,case,xL,xR,N,t0,T,Mthdname,rkim,rkex,b,c,dt):  
    # spatial discretization
    L = xR-xL; x = np.linspace(xL, xR, N+1)[:-1] # exclude the right boundary point
    dx = x[1]-x[0]; xi = np.fft.fftfreq(N) * N * 2*np.pi / L
    
    u0 = IC(x,ep,case); uu = np.zeros((1,np.size(u0))); uu[0,:] = u0.copy()
    tt = np.zeros(1); t = t0; tt[0] = t
    
    s = len(rkim); Rim = np.zeros((s,len(u0))); Rex = np.zeros((s,len(u0))); steps = 0  
    
    # time loop
    t_start = time()
    while t < T and not np.isclose(t, T):
        if t + dt > T:
            dt = T - t
        for i in range(s):
            rhs = uu[-1].copy()
            if i>0:
                for j in range(i):
                    rhs += dt*(rkim[i,j]*Rim[j,:] + rkex[i,j]*Rex[j,:])

            Coeff =1/(1-dt*rkim[i,i]*(-1j*ep*xi*xi/2))
            g_j_comp = np.fft.ifft(Coeff*np.fft.fft(rhs[:N]+1j*rhs[N:]))
            g_j = np.concatenate((g_j_comp.real,g_j_comp.imag)).astype('float64')
            Rim[i,:] = f_stiff(g_j,N,xi,ep)
            Rex[i,:] = f_non_stiff(g_j,N,ep,case)

        inc = dt*sum([ b[j]*(Rim[j]+Rex[j]) for j in range(s)])    
        unew = uu[-1]+inc; t+= dt
        tt = np.append(tt, t)
        steps += 1
        uu = np.append(uu, np.reshape(unew.copy(), (1,len(unew))), axis=0) 
        
    t_end = time()
    req_t = t_end-t_start     
    return req_t, tt, uu

# This is an implementation of an ImEx time integrator with multi-relaxation
def Compute_Sol_ImEx_Relaxation(f_stiff,f_non_stiff,IC,ep,case,xL,xR,N,t0,T,Mthdname,rkim,rkex,b,bhat,c,inv,eta1,eta2,rgam,dt): 
    # spatial discretization
    L = xR-xL; x = np.linspace(xL, xR, N+1)[:-1] # exclude the right boundary point
    dx = x[1]-x[0]; xi = np.fft.fftfreq(N) * N * 2*np.pi / L
    
    tt = np.zeros(1); t = t0; tt[0] = t # time
    u0 = IC(x,ep,case); uu = np.zeros((1,np.size(u0))); uu[0,:] = u0.copy() # solution
    
    s = len(rkim); Rim = np.zeros((s,len(u0))); Rex = np.zeros((s,len(u0))) 
    gamma0 = np.zeros(inv); G = np.zeros((1,np.size(gamma0))); G[0,:] = gamma0.copy()
    steps = 0; no_ier_five = 0; no_ier_one = 0; no_ier_else = 0
    t_start = time()
    while t < T and not np.isclose(t, T):
        if t + dt > T:
            dt = T - t        
        for i in range(s):
            rhs = uu[-1].copy()
            if i>0:
                for j in range(i):
                    rhs += dt*(rkim[i,j]*Rim[j,:] + rkex[i,j]*Rex[j,:] )
                    
                    
            Coeff =1/(1-dt*rkim[i,i]*(-1j*ep*xi*xi/2))
            g_j_comp = np.fft.ifft(Coeff*np.fft.fft(rhs[:N]+1j*rhs[N:]))
            g_j = np.concatenate((g_j_comp.real,g_j_comp.imag)).astype('float64')
            Rim[i,:] = f_stiff(g_j,N,xi,ep)
            Rex[i,:] = f_non_stiff(g_j,N,ep,case)
        
        inc1 = dt*sum([ b[i]*(Rim[i]+Rex[i]) for i in range(s)])  
        inc2 = dt*sum([ bhat[i]*(Rim[i]+Rex[i]) for i in range(s)]) 
        unew = uu[-1]+inc1; 
        
        if inv == 1:
            inc = np.array([inc1]); E_old = np.array([eta1(uu[-1],N,dx)])
        elif inv == 2:
            inc = np.array([inc1,inc2]); E_old = np.array([eta1(uu[-1],N,dx), eta2(uu[-1],N,dx,ep,case)])
                
        # fsolve
        ga_fsolve, info, ier, mesg = fsolve(rgam,gamma0,args=(unew,N,dx,ep,case,inc,inv,E_old),full_output=True)
        gamma = ga_fsolve; gamma0 = ga_fsolve                    

        unew = unew + np.dot(gamma,inc); t+=(1+sum(gamma))*dt
        tt = np.append(tt, t); uu = np.append(uu, np.reshape(unew.copy(), (1,len(unew))), axis=0)  
    t_end = time()
    req_t = t_end-t_start
    return req_t,tt, uu, G, no_ier_one, no_ier_five, no_ier_else


def adaptive_step_local_err(u1,u2,dt,tol,em_or):
    fac = 0.9; m = len(u1); 
    vec = (u1-u2)/(tol*(1+np.max(np.array([np.abs(u1),np.abs(u2)]),0)))
    w_n_1 = np.sqrt(sum(vec**2)/m)
    ep_n_1 = 1/w_n_1
    mult_fac = fac*np.power(ep_n_1,1/(em_or+1))
    dt_new = mult_fac*dt
    return mult_fac, dt_new

# This is an implementation of an ImEx time integrator with multi-relaxation and adaptive step size control
def Compute_Sol_ImEx_Relaxation_Adp(f_stiff,f_non_stiff,IC,ep,case,xL,xR,N,t0,T,Mthdname,rkim,rkex,b,bhat,c,inv,eta1,eta2,rgam,dt,em_or,tol): 
    # to store variable tile steps
    dt0 = dt; dts = np.zeros(1); dts[0] = dt
    
    # spatial discretization
    L = xR-xL; x = np.linspace(xL, xR, N+1)[:-1] # exclude the right boundary point
    dx = x[1]-x[0]; xi = np.fft.fftfreq(N) * N * 2*np.pi / L
    
    tt = np.zeros(1); t = t0; tt[0] = t # time
    u0 = IC(x,ep,case); uu = np.zeros((1,np.size(u0))); uu[0,:] = u0.copy() # solution
    
    s = len(rkim); Rim = np.zeros((s,len(u0))); Rex = np.zeros((s,len(u0))) 
    gamma0 = np.zeros(inv); G = np.zeros((1,np.size(gamma0))); G[0,:] = gamma0.copy()
    steps = 0; no_ier_five = 0; no_ier_one = 0; no_ier_else = 0

    # time loop
    t_start = time()
    while t < T and not np.isclose(t, T):
        if t + dt > T:
            dt = T - t        
        for i in range(s):
            rhs = uu[-1].copy()
            if i>0:
                for j in range(i):
                    rhs += dt*(rkim[i,j]*Rim[j,:] + rkex[i,j]*Rex[j,:] )
                    
            Coeff =1/(1-dt*rkim[i,i]*(-1j*ep*xi*xi/2))
            g_j_comp = np.fft.ifft(Coeff*np.fft.fft(rhs[:N]+1j*rhs[N:]))
            g_j = np.concatenate((g_j_comp.real,g_j_comp.imag)).astype('float64')
            Rim[i,:] = f_stiff(g_j,N,xi,ep)
            Rex[i,:] = f_non_stiff(g_j,N,ep,case)
        
        inc1 = dt*sum([ b[i]*(Rim[i]+Rex[i]) for i in range(s)])  
        inc2 = dt*sum([ bhat[i]*(Rim[i]+Rex[i]) for i in range(s)])
        unew = uu[-1]+inc1; unew_em = uu[-1]+inc2;
        
        mult_fac, dt_new = adaptive_step_local_err(unew,unew_em,dt,tol,em_or)
        if mult_fac >= 0.9:
            if inv == 1:
                inc = np.array([inc1]); E_old = np.array([eta1(uu[-1],N,dx)])
            elif inv == 2:
                inc = np.array([inc1,inc2]); E_old = np.array([eta1(uu[-1],N,dx), eta2(uu[-1],N,dx,ep,case)])        
            # fsolve
            ga_fsolve, info, ier, mesg = fsolve(rgam,gamma0,args=(unew,N,dx,ep,case,inc,inv,E_old),full_output=True)
            gamma = ga_fsolve; gamma0 = ga_fsolve 
            Err = np.linalg.norm(info['fvec'])
            if Err >= 1e-12:
                dt = dt/2
            else:
                dts = np.append(dts,dt)
                # relaxation solution
                unew = unew + np.dot(gamma,inc); t+=(1+sum(gamma))*dt
                tt = np.append(tt, t); uu = np.append(uu, np.reshape(unew.copy(), (1,len(unew))), axis=0)  
                #G = np.append(G, np.reshape(gamma.copy(), (1,len(gamma))), axis=0) 
                dt = dt_new # update time step for next time step
        else: 
            dt = dt_new # local error control fails, reject the step and recompute with controller predicted step
            #print('Modified dt by step size control = %1.4f \n'%dt)
    t_end = time()
    req_t = t_end-t_start
    return req_t, tt, uu, G, no_ier_one, no_ier_five, no_ier_else

# This is an implementation of an operator splitting 
# substeps
def sol_nl_part(nonlin_f,ep, case, u, dt):
    u_st = np.exp((-1j/ep)*nonlin_f(np.abs(u)**2,ep,case)*dt)*u
    return u_st
              
def sol_l_part(ep, u, xi, dt):
    uhat = np.fft.fft(u)
    u_st_hat = np.exp(dt*(-1j*ep*xi**2)/2)*uhat
    u_st = np.fft.ifft(u_st_hat)
    return u_st

def Op_Split_Exact_Solve(nonlin_f,sol_nl_part,sol_l_part,IC,ep,case,xL,xR,N,t0,T,Mthdname,a,b,dt):
    # spatial discretization
    L = xR-xL; x = np.linspace(xL, xR, N+1)[:-1] # exclude the right boundary point
    dx = x[1]-x[0]; xi = np.fft.fftfreq(N) * N * 2*np.pi / L
    
    tt = np.zeros(1); t = t0; tt[0] = t # time
    u0 = IC(x,ep,case); uu = np.zeros((1,np.size(u0))); uu[0,:] = u0.copy() # solution
    u_comp = u0[:N]+ (1j) *u0[N:]; steps = 0
    t_start = time()
    while t < T and not np.isclose(t, T):
        if t + dt > T:
            dt = T - t
        u_st = u_comp.copy()  
        for j in range(len(a)):
            u_st = sol_nl_part(nonlin_f,ep,case,u_st,a[j]*dt)
            u_st = sol_l_part(ep,u_st,xi,b[j]*dt)
            
        u_comp = u_st.copy(); t+=dt
            
        unew = np.concatenate((u_comp.real,u_comp.imag)).astype('float64')
        tt = np.append(tt, t); uu = np.append(uu, np.reshape(unew.copy(), (1,len(unew))), axis=0)  
        steps += 1   
    t_end = time()
    req_t = t_end-t_start
    return req_t,tt,uu

# ImEx methods
def ImEx_schemes(s,p,emp,sch_no):
    #3rd order ImEx with b and 2nd order ImEx with bhat. This method is taken from Implicit-explicit 
    # Runge-Kutta methods for time-dependent partial differential equations by Ascher, Steven Ruuth and Spiteri.
    if s == 4 and p == 3 and emp == 2 and sch_no == 2:
        rkim = np.array([ [ 0,              0,             0,             0],
                          [ 0,   0.4358665215,             0,             0],
                          [ 0,   0.2820667392,  0.4358665215,             0],
                          [ 0,   1.2084966490, -0.6443631710,  0.4358665215] ])
        rkex = np.array([ [ 0,                       0,             0,             0],
                          [ 0.4358665215,            0,             0,             0],
                          [ 0.3212788860, 0.3966543747,             0,             0],
                          [-0.1058582960, 0.5529291479,  0.5529291479,             0] ])
        c = sum(rkex.T)
        b = np.array([0,   1.2084966490, -0.6443631710,  0.4358665215])
        bhat = np.array([0,0.886315063820486,0 ,0.113684936179514])
    #3rd order ImEx with b and 2nd order ImEx with bhat. This method is taken from Additive Runge–Kutta schemes 
    #for convection–diffusion–reaction equations by Kennedy and Carpenter.
    if s == 4 and p == 3 and emp == 2 and sch_no == 3:
        rkim = np.array([ [ 0,              0,             0,             0],
                  [1767732205903/4055673282236, 1767732205903/4055673282236, 0, 0],
                  [2746238789719/10658868560708, -640167445237/6845629431997, 1767732205903/4055673282236, 0],              
                  [1471266399579/7840856788654, -4482444167858/7529755066697, 11266239266428/11593286722821,                 1767732205903/4055673282236] ])

        rkex = np.array([ [0,                            0,         0,             0],
                          [1767732205903/2027836641118,  0,         0,             0],
                          [5535828885825/10492691773637, 788022342437/10882634858940, 0, 0],
                          [6485989280629/16251701735622, -4246266847089/9704473918619, 10755448449292/10357097424841, 0] ])

        c = np.array([0, 1767732205903/2027836641118, 3/5, 1])
        b = np.array([1471266399579/7840856788654, -4482444167858/7529755066697, 11266239266428/11593286722821, 1767732205903/4055673282236])
        bhat = np.array([2756255671327/12835298489170, -10771552573575/22201958757719, 9247589265047/10645013368117, 2193209047091/5459859503100])

    #4th order ImEx with b and 3rd order ImEx with bhat. This method is taken from Additive Runge–Kutta schemes 
    #for convection–diffusion–reaction equations by Kennedy and Carpenter.
    elif s == 6 and p == 4 and emp == 3 and sch_no == 4:
        rkex = np.array([ [0, 0, 0, 0, 0, 0],
                  [1/2, 0, 0, 0, 0, 0],
                  [13861/62500, 6889/62500, 0, 0, 0, 0], 
                  [-116923316275/2393684061468, -2731218467317/15368042101831, 9408046702089/11113171139209, 0, 0, 0], 
                  [-451086348788/2902428689909, -2682348792572/7519795681897, 12662868775082/11960479115383, 3355817975965/11060851509271, 0, 0], 
                  [647845179188/3216320057751, 73281519250/8382639484533, 552539513391/3454668386233, 3354512671639/8306763924573, 4040/17871, 0] 
                ])

        rkim = np.array([ [0, 0, 0, 0, 0, 0],
                          [1/4, 1/4, 0, 0, 0, 0],
                          [8611/62500, -1743/31250, 1/4, 0, 0, 0],
                          [5012029/34652500, -654441/2922500, 174375/388108, 1/4, 0, 0],
                          [15267082809/155376265600, -71443401/120774400, 730878875/902184768, 2285395/8070912, 1/4, 0],
                          [82889/524892, 0, 15625/83664, 69875/102672, -2260/8211, 1/4]
                        ])

        c = np.array([0, 1/2, 83/250, 31/50, 17/20, 1])
        b = np.array([82889/524892, 0, 15625/83664, 69875/102672, -2260/8211, 1/4])
        bhat = np.array([4586570599/29645900160, 0, 178811875/945068544, 814220225/1159782912, -3700637/11593932, 61727/225920])
        
    return rkim, rkex, c, b, bhat 

# Operator splitting methods
def Op_Sp_Coeff(s,p,sch_no):
    # A 1-stage 1st order operator splitting method: Lie-Trotter
    if s == 1 and p == 1 and sch_no == 0:      
        a = np.array([1.])
        b = np.array([1.])
    # A 2-stage 2nd order operator splitting method: Strang splitting
    elif s == 2 and p == 2 and sch_no == 1:  
        a = np.array([1/2,1/2])
        b = np.array([1,0.])
    # A 5-stage 4th order operator splitting method
    elif s == 5 and p == 4 and sch_no == 2:
        a = np.array([0.267171359000977615,-0.0338279096695056672,
                      0.5333131013370561044,-0.0338279096695056672
                      ,0.267171359000977615])
        b = np.array([-0.361837907604416033,0.861837907604416033,
                      0.861837907604416033,-0.361837907604416033,0.])
        
    return a, b

