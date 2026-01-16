import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import util
import os
import engutil

def loudspeaker_ode(t, x, u_func, params, polys):
    i_curr = x[0]  
    i_creep = x[1] 
    disp = x[2]    
    vel = x[3]    
    
    voltage = u_func(t) # input voltage
    
    # values from Bl, K and Le polynomials
    val_Bl = polys['Bl'](disp)
    val_K  = polys['K'](disp)
    val_Le_base = polys['Le'](disp) * polys['Li'](disp)
    val_Le = max(val_Le_base, 1e-9) # ensure there will be no divide by 0

    Le_nominal = polys['Le'](0) # x = 0 value of Le
    ratio = val_Le / Le_nominal
    
    # scale R2 and L2 accordingly
    val_R2 = params['R20'] * ratio
    val_L2 = params['L20'] * ratio
    
    # equation 0: di/dt = (1/Le) * (V - (Re+R2)i + R2*i2 - Bl*v)
    di_dt = (1.0 / val_Le) * (voltage - (params['Re'] + val_R2)*i_curr + val_R2*i_creep - val_Bl*vel)
    
    # equation 1: di2/dt = (1/L2) * (R2*i - R2*i2)
    di2_dt = (val_R2 / val_L2) * (i_curr - i_creep)
    
    # equation 2: d(disp)/dt = vel
    ddisp_dt = vel
    
    # equation 3: d(vel)/dt = (1/Mm) * (Bl*i - K*x - Rm*v)
    dvel_dt = (1.0 / params['Mm']) * (val_Bl*i_curr - val_K*disp - params['Rm']*vel)
    
    return [di_dt, di2_dt, ddisp_dt, dvel_dt]

def solve_nonlinear_euler(u_signal, x0, fs, params, polys):
    """
    params = {
    'Re':  R_e,
    'Rm':  R_m,
    'Mm':  M_m,
    'R20': R_20,
    'L20': L_20

}

polys = {
    'Bl': poly_Bl,
    'K':  poly_K,
    'Le': poly_Le,
    'Li': poly_Li
}

where poly_Bl =poly_Bl = np.poly1d(Bln)

    """
    Ts = 1.0 / fs
    num_steps = len(u_signal)
    num_states = len(x0)
    
    # output:
    x_history = np.zeros((num_steps, num_states))
    x_curr = x0.copy()
    
    # static values
    Re = params['Re']
    R20 = params['R20']
    L20 = params['L20']
    Mm = params['Mm']
    Rm = params['Rm']
    
    # init matrices
    F = np.zeros((4, 4))
    G = np.zeros(4)
    
    # static parts of F
    
    F[2, 3] = 1.0
    F[3, 3] = -Rm / Mm
    
    for n in range(num_steps):
        x_history[n] = x_curr
        
        # i, i_L2, displacement, velocity
        i_curr = x_curr[0]
        disp   = x_curr[2] # grab displacement value
        
        # update polynomial values
        val_Bl = polys['Bl'](disp)
        val_K  = polys['K'](disp)
        val_Le = polys['Le'](disp)*polys["Li"](disp)
        val_R2 = R_20*val_Le/polys['Le'](0)
        val_L2 = L_20*val_Le/polys['Le'](0)

        # L_e = Ln*Li 
        # R_2 = R_20*L_e/L_e0
        # L_2 = L_20*L_e/L_e0

        # val_L2 = polys['Li'](disp)

        # val_Le = max(val_Le, 1e-6) 
        # val_L2 = max(val_Li, 1e-6)
        
        # # update F whereever it needs updating
        # F[0, 0] = -(Re + R20) / val_Le
        # F[0, 1] = R20 / val_Le
        # F[0, 3] = -val_Bl / val_Le
        
        # F[1, 0] = R20 / val_L2
        # F[1, 1] = -R20 / val_L2

        # F[3, 0] = val_Bl / Mm
        # F[3, 2] = -val_K / Mm


        F[0, 0] = -(Re + val_R2) / val_Le
        F[0, 1] = val_R2 / val_Le
        F[0, 3] = -val_Bl / val_Le
        
        F[1, 0] = val_R2 / val_L2
        F[1, 1] = -val_R2 / val_L2

        F[3, 0] = val_Bl / Mm
        F[3, 2] = -val_K / Mm



        G[0] = 1.0 / val_Le
        
        # do the Euler!
        dx = (F @ x_curr) + (G * u_signal[n])
        
        x_next = x_curr + (dx * Ts)
        x_curr = x_next
        
    return x_history


