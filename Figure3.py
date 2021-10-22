"""
Code underlying Figure 3 of the manuscript
'Principles underlying the complex dynamics of
temperature entrainment by a circadian clock'

For three exemplary conditions, we simulate 
the dynamics of a temperature entrained model
of the Neurospora circadian clock.
"""


from pylab import*
from scipy.signal import hilbert
from scipy.integrate import odeint
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
import pickle
import itertools
import multiprocessing
import os

current_palette = sns.color_palette() 

#==============================================================================
# parameter dictionaries
#==============================================================================

# rate constants per hour for the WT strain
rate = {
    'k1'    : 1.8,
    'k2'    : 1.8,
    'k3'    : 0.05,
    'k4'    : 0.23,
    'k5'    : 0.27,
    'k6'    : 0.07,
    'k7'    : 0.5,
    'k8'    : 0.8,
    'k9'    : 40.0,
    'k10'   : 0.3,
    'k11'   : 0.05,
    'k12'   : 0.02,
    'k13'   : 50.0,
    'k14'   : 1.0,
    'k15'   : 8.0,
    'K'     : 1.25,
    'K2'    : 1.0
}

# rate constants per hour for the frq1 mutant
rate1 = {
    'k1'    : 1.8,
    'k2'    : 1.8,
    'k3'    : 0.15,
    'k4'    : 0.23,
    'k5'    : 0.4,
    'k6'    : 0.1,
    'k7'    : 0.5,
    'k8'    : 0.8,
    'k9'    : 40.0,
    'k10'   : 0.3,
    'k11'   : 0.05,
    'k12'   : 0.02,
    'k13'   : 50.0,
    'k14'   : 1.0,
    'k15'   : 8.0,
    'K'     : 1.25,
    'K2'    : 1.0
}

# rate constants per hour for frq7 mutant
rate7 = {
    'k1'    : 1.8,
    'k2'    : 1.8,
    'k3'    : 0.05,
    'k4'    : 0.23,
    'k5'    : 0.15,
    'k6'    : 0.01,
    'k7'    : 0.5,
    'k8'    : 0.8,
    'k9'    : 40.0,
    'k10'   : 0.3,
    'k11'   : 0.05,
    'k12'   : 0.02,
    'k13'   : 50.0,
    'k14'   : 1.0,
    'k15'   : 8.0,
    'K'     : 1.25,
    'K2'    : 1.0
}

# function to set parameter values of the clock model within the extension
# module "mod_name", taken from the parameter dictonary

def SetOscillatorParameters(c_rates):
    mod_name.c5  = c_rates["k1"]
    mod_name.c6  = c_rates["k2"]
    mod_name.c7  = c_rates["k3"]
    mod_name.c8  = c_rates["k4"]
    mod_name.c9  = c_rates["k5"]
    mod_name.c10 = c_rates["k6"]
    mod_name.c11 = c_rates["k7"]
    mod_name.c12 = c_rates["k8"]
    mod_name.c13 = c_rates["k9"]
    mod_name.c14 = c_rates["k10"]
    mod_name.c15 = c_rates["k11"]
    mod_name.c16 = c_rates["k12"]
    mod_name.c17 = c_rates["k13"]
    mod_name.c18 = c_rates["k14"]
    mod_name.c19 = c_rates["k15"]
    mod_name.c20 = c_rates["K"]
    mod_name.c21 = c_rates["K2"]

# initial conditions
frq_mrna0    = 4.0
frq_c0       = 30.0
frq_n0       = 0.1
wc1_mrna0    = (0.5 / 0.3)
wc1_c0       = 0.03225
wc1_n0       = 0.35
frq_n_wc1_n0 = 0.18

state0 = [frq_mrna0,
          frq_c0,
          frq_n0,
          wc1_mrna0,
          wc1_c0,
          wc1_n0,
          frq_n_wc1_n0]


# functions to test for entrainment

def T_detection_epsilon_ball(solution_array, t_trans, neighborhood_epsilon, t_step):
    save_initial = solution_array[int(t_trans/t_step)].copy()
    distance_from_inital_state = sqrt(sum((solution_array - save_initial)**2, axis=1))
    points_inside_epsilon_ball_indices = argwhere( distance_from_inital_state <= neighborhood_epsilon ).flatten()
    leftmost_points_in_epsilon_ball = points_inside_epsilon_ball_indices[argwhere(points_inside_epsilon_ball_indices - roll(points_inside_epsilon_ball_indices, 1) != 1)].flatten()
    #period_array = (leftmost_points_in_epsilon_ball[1:] - leftmost_points_in_epsilon_ball[:-1])*t_step
    period_array = (leftmost_points_in_epsilon_ball[2:] - leftmost_points_in_epsilon_ball[1:-1])*t_step      # neglect 1st period
    return period_array

def numpy_simple_shooting(sol_vector, entrainment_cycles, t_trans_cycles, t_period, t_step, neighborhood_epsilon):
    save_initial = copy(sol_vector[int((t_trans_cycles)*t_period/t_step)])
    test_set =  sol_vector[array((t_trans_cycles + array(range(entrainment_cycles - t_trans_cycles)))*t_period/t_step, dtype=int)]
    c_x = argwhere(sqrt(sum((test_set - save_initial)**2, axis=1)) <= neighborhood_epsilon).flatten()
    return c_x[1:] - c_x[:-1]



# import extensiom module containing the Neurospora clock model
import ForcedHongModel as ForcedHongModel
mod_name = ForcedHongModel.mod_forced_hong_model


# set simulation parameters
kappa       = 0.5           # thermoperiod
s           = 100           # steepness of the ZG function
mod_name.c1 = copy(kappa)   # Zeitgeber photo/thermoperiod
mod_name.c3 = copy(s)       # Zeitgeber steepness
mod_name.c4 = 0             # Phase shift of ZG in hours
mod_name.c22 = 2.           # Hill Coefficient

# set initial conditions
X0 = [1., 0] + state0
SetOscillatorParameters(rate)

# set numerical parameters
entrainment_cycles = 520
t_trans_cycles = entrainment_cycles - 20
neighborhood_epsilon = 0.1
t_step = 0.1

T_steps = 0.1
Z_steps = 0.005
T_min, T_max = 8., 28. +T_steps
Z_min, Z_max = 0, 1.+Z_steps
T_2_mesh, Z_2_mesh = arange(T_min, T_max, T_steps), arange(Z_min, Z_max, Z_steps)
T_vec, Z_vec = meshgrid(T_2_mesh, Z_2_mesh)
paramlist = list(itertools.product(T_2_mesh,Z_2_mesh))

# define function to calculate entrainment properties
def CalculateEntrainmentPropertiesF2py(params):
    T = params[0]
    Z = params[1]
    mod_name.c0 = copy(T)
    mod_name.c2 = copy(Z)
    try:
        t = arange(0, T*entrainment_cycles, t_step)
        c_hmax = (1.-(kappa))*T/2
        sol = odeint(mod_name.ode, X0, t)
        determine_per = T_detection_epsilon_ball(sol, t_trans_cycles, neighborhood_epsilon, t_step)
        c_a, c_b = mean(determine_per), std(determine_per)
        try:
            determine_per_shooting = numpy_simple_shooting(sol, entrainment_cycles, t_trans_cycles, T, t_step, neighborhood_epsilon)
            c_c, c_d = mean(determine_per_shooting), std(determine_per_shooting)
        except:
            c_c, c_d = 0., 0.
    except:
        c_a, c_b = 0., 0.
    return c_a, c_b, c_c, c_d


# Check if Arnold tongue has been already calculated. If not do the calculation - if yes, load the simulation data
SavePath = "./ArnoldOnion/ArnoldZmin"+str(Z_min)+"Zmax"+str(Z_max)+"Zstep"+str(Z_steps)+"Tmin"+str(T_min)+"Tmax"+str(T_max)+"Tstep"+str(T_steps)+".npz"
if os.path.exists(SavePath) != True:
    print "Doing Calculation"
    pool = multiprocessing.Pool()
    res  = pool.map(CalculateEntrainmentPropertiesF2py,paramlist)
    SaveData = (T_2_mesh, Z_2_mesh, res)
    np.savez_compressed(SavePath, SaveData)
else:
    LoadResults = np.load(SavePath, allow_pickle=True)["arr_0"]
    T_2_mesh, Z_2_mesh, res = LoadResults
    print "Loaded"
    

# select 1:1 and 1:2 entrainment regions
c_res_mean = array(res).T[2]
c_res_std = array(res).T[3]
c_res_mean[argwhere(c_res_std > 0.1)] = 0

Arnold1to1 = np.ma.masked_where(( c_res_mean != 1) , c_res_mean)
Arnold1to2 = np.ma.masked_where(( c_res_mean != 2) , c_res_mean+1)


# Arnold tongue figure
figure(figsize=(6, 3.5))
c_snap = False
c_colormap = "tab10"
p1 = pcolor(T_vec, Z_vec, Arnold1to1.reshape(len(T_2_mesh),len(Z_2_mesh)).T, snap=c_snap, label="1:1", linewidth=0, vmin=1, vmax=11, cmap=c_colormap, rasterized=True)
p1.set_edgecolor('face')
p2 = pcolor(T_vec, Z_vec, Arnold1to2.reshape(len(T_2_mesh),len(Z_2_mesh)).T, snap=c_snap, label="1:2", linewidth=0, vmin=1, vmax=11, cmap=c_colormap, rasterized=True)
p2.set_edgecolor('face')

c_Z = 0.45
c_T1 = 22
c_T2 = 16
c_T3 = 12
gray_star = mlines.Line2D([], [], color='k', marker='*', linestyle='None', markersize=10, label='T='+str(c_T1)+'h')
gray_square = mlines.Line2D([], [], color='k', marker='s', linestyle='None', markersize=10, label='T='+str(c_T2)+'h')
gray_triangle = mlines.Line2D([], [], color='k', marker='^', linestyle='None', markersize=10, label='T='+str(c_T3)+'h')

scatter(c_T1, c_Z, marker="*", s=40, color="k")
scatter(c_T2, c_Z, marker="s", s=40, color="k")
scatter(c_T3, c_Z, marker="^", s=40, color="k")

xticks(arange(10,28, 4), fontsize=15)
xlim(10, 28)
yticks([0, 0.5, 1.], fontsize=15)
ylim(0, 1)
xlabel("T [h]", fontsize=15)
ylabel("Z [a.u.]", fontsize=15)
title("C", loc="left", fontsize=20)
tight_layout()


# plot example simularions
figure(figsize=(15, 14.5/2))
plot_cycles = 10

# set simulation parameters 
SetOscillatorParameters(rate)
T = c_T1
Z = c_Z
kappa = 0.5
mod_name.c0 = copy(T)                       # ZG period
mod_name.c1 = copy(kappa)                   # ZG thermoperiod
mod_name.c2 = copy(Z)                       # ZG amplitude
mod_name.c4 = (1.-kappa)*T/2+2.*T*kappa	    # Phase shift of ZG in hours
t = arange(0, T*entrainment_cycles, t_step)
c_hmax = (1.-(kappa))*T/2

# solve equations
sol = odeint(mod_name.ode, X0, t)

# plot soluations
subplot(2,3,1)
plot(t[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], sol.T[2][int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)])
coeff=mod_name.c3*mod_name.c0**2/(2*pi*(mod_name.c1*mod_name.c0))
arg=sol.T[0]*cos(2.*pi*mod_name.c4/mod_name.c0)+sol.T[1]*sin(2.*pi*mod_name.c4/mod_name.c0)-cos(pi*mod_name.c1)
force=mod_name.c2*(arctan(coeff*arg)/pi)
fill_between(t[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], 0, 20*(Z/2-force)[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], color="lightgray")
xticks(arange(T*t_trans_cycles, T*t_trans_cycles+9.*24, 24), arange(0, 9*24, 24), fontsize=15)
yticks([0, 4, 8], fontsize=15)
xlim(T*t_trans_cycles, T*t_trans_cycles+8.*24)
ylim(0, 8)
xticks()
title("D", loc="left", fontsize=20)
xlabel("Time [h]", fontsize=15)
ylabel(r"$frq$ mRNA [a.u.]", fontsize=15)
legend(handles=[gray_star], loc="upper right")

# plot instantaneous phase of solutions versus ZG phase
subplot(2,3,4)
detrended_signal_frq    = sol.T[2] - mean(sol.T[2][int(T*t_trans_cycles / t_step):])
hilbert_tranformed_frq  = hilbert(detrended_signal_frq)
angle_frq               = np.angle(hilbert_tranformed_frq) 
ampli_frq               = np.abs(hilbert_tranformed_frq) 
scatter(((2.*pi/T*t)%(2.*pi)-pi)[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], angle_frq[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], alpha=0.5, edgecolors=None, s=10, color=current_palette[0])
xlim(-pi, pi)
ylim(-pi, pi)
xlabel("Zeitgeber Phase [rad]", fontsize=15)
ylabel("$frq$ mRNA Phase [rad]", fontsize=15)
xticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"], fontsize=15)
yticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"], fontsize=15)
title("G", fontsize=20, loc="left")
legend(handles=[gray_star], loc="upper right")


subplot(2, 3, 2)
#simulation parameters
T = c_T2
Z = c_Z
mod_name.c0 = copy(T)
mod_name.c2 = copy(Z)
mod_name.c4 = (1.-kappa)*T/2+2.*T*kappa
t = arange(0, T*entrainment_cycles, t_step)
c_hmax = (1.-(kappa))*T/2

# solve equations
sol = odeint(mod_name.ode, X0, t)

# plot soluations
plot(t[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], sol.T[2][int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)])
coeff=mod_name.c3*mod_name.c0**2/(2*pi*(mod_name.c1*mod_name.c0))
arg=sol.T[0]*cos(2.*pi*mod_name.c4/mod_name.c0)+sol.T[1]*sin(2.*pi*mod_name.c4/mod_name.c0)-cos(pi*mod_name.c1)
force=mod_name.c2*(arctan(coeff*arg)/pi)
fill_between(t[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], 0, 20*(Z/2-force)[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], color="lightgray")
xlim(T*t_trans_cycles, T*t_trans_cycles+8.*24)
xticks(arange(T*t_trans_cycles, T*t_trans_cycles+9.*24, 24), arange(0, 9*24, 24), fontsize=15)
yticks([0, 4, 8], fontsize=15)
xlim(T*t_trans_cycles, T*t_trans_cycles+8.*24)
ylim(0, 8)
title("E", loc="left", fontsize=20)
xlabel("Time [h]", fontsize=15)
legend(handles=[gray_square], loc="upper right")

# plot instantaneous phase of solutions versus ZG phase
subplot(2, 3, 5)
detrended_signal_frq    = sol.T[2] - mean(sol.T[2][int(T*t_trans_cycles / t_step):])
hilbert_tranformed_frq  = hilbert(detrended_signal_frq)
angle_frq               = np.angle(hilbert_tranformed_frq) 
ampli_frq               = np.abs(hilbert_tranformed_frq) 
scatter(((2.*pi/T*t)%(2.*pi)-pi)[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], angle_frq[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], alpha=0.5, edgecolors=None, s=10, color=current_palette[0])
xlim(-pi, pi)
ylim(-pi, pi)
xlabel("Zeitgeber Phase [rad]", fontsize=15)
xticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"], fontsize=15)
yticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"], fontsize=15)
title("H", fontsize=20, loc="left")
legend(handles=[gray_square], loc="upper right")

subplot(2, 3, 3)
#simulation parameters
T = c_T3
Z = c_Z
mod_name.c0 = copy(T)
mod_name.c2 = copy(Z)
mod_name.c4 = (1.-kappa)*T/2+2.*T*kappa		# Phase shift of ZG in hours
t = arange(0, T*entrainment_cycles, t_step)
c_hmax = (1.-(kappa))*T/2

# solve equations
sol = odeint(mod_name.ode, X0, t)

# plot soluations
plot(t[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], sol.T[2][int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)])
coeff=mod_name.c3*mod_name.c0**2/(2*pi*(mod_name.c1*mod_name.c0))
arg=sol.T[0]*cos(2.*pi*mod_name.c4/mod_name.c0)+sol.T[1]*sin(2.*pi*mod_name.c4/mod_name.c0)-cos(pi*mod_name.c1)
force=mod_name.c2*(arctan(coeff*arg)/pi)
fill_between(t[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], 0, 20*(Z/2-force)[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], color="lightgray")
xlim(T*t_trans_cycles, T*t_trans_cycles+8.*24)
xticks(arange(T*t_trans_cycles, T*t_trans_cycles+9.*24, 24), arange(0, 9*24, 24), fontsize=15)
yticks([0, 4, 8], fontsize=15)
xlim(T*t_trans_cycles, T*t_trans_cycles+8.*24)
ylim(0, 8)
xlabel("Time [h]", fontsize=20)
title("F", loc="left", fontsize=20)
legend(handles=[gray_triangle], loc="upper right")

# plot instantaneous phase of solutions versus ZG phase
subplot(2, 3, 6)
detrended_signal_frq    = sol.T[2] - mean(sol.T[2][int(T*t_trans_cycles / t_step):])
hilbert_tranformed_frq  = hilbert(detrended_signal_frq)
angle_frq               = np.angle(hilbert_tranformed_frq) 
ampli_frq               = np.abs(hilbert_tranformed_frq) 
scatter(((2.*pi/T*t)%(2.*pi)-pi)[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], angle_frq[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)], alpha=0.5, edgecolors=None, s=10, color=current_palette[0])
xlim(-pi, pi)
ylim(-pi, pi)
xlabel("Zeitgeber Phase [rad]", fontsize=15)
xticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"], fontsize=15)
yticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"], fontsize=15)
title("I", fontsize=20, loc="left")
legend(handles=[gray_triangle], loc="upper right")



tight_layout()

show()


