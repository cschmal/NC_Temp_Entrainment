"""
Code underlying the calculation fitting scores
as plotted in Figure 5B of the manuscript
'Principles underlying the complex dynamics of
temperature entrainment by a circadian clock'
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.collections as collections
import string
from pylab import*
import seaborn as sns
from scipy.signal import hilbert
import pickle
import itertools
import multiprocessing
import os


#==============================================================================
# parameter dictionaries
#==============================================================================
### rate constants per hour
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

def SetOscillatorParametersScalePeriod(c_rates, c_scale):
    mod_name.c5  = c_rates["k1"] * c_scale
    mod_name.c6  = c_rates["k2"] * c_scale
    mod_name.c7  = c_rates["k3"] * c_scale
    mod_name.c8  = c_rates["k4"] * c_scale
    mod_name.c9  = c_rates["k5"] * c_scale
    mod_name.c10 = c_rates["k6"] * c_scale
    mod_name.c11 = c_rates["k7"] * c_scale
    mod_name.c12 = c_rates["k8"] * c_scale
    mod_name.c13 = c_rates["k9"] * c_scale
    mod_name.c14 = c_rates["k10"] * c_scale
    mod_name.c15 = c_rates["k11"] * c_scale
    mod_name.c16 = c_rates["k12"] * c_scale
    mod_name.c17 = c_rates["k13"] * c_scale
    mod_name.c18 = c_rates["k14"] * c_scale
    mod_name.c19 = c_rates["k15"] * c_scale
    mod_name.c20 = c_rates["K"]
    mod_name.c21 = c_rates["K2"]


#==============================================================================
# initial conditions
#==============================================================================
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



T     = 12.
kappa = 0.5
warm_dur = T*(1-kappa)
s=100
z0= 0.15

entrainment_cycles = 520
#entrainment_cycles = 120
t_trans_cycles = entrainment_cycles - 20
neighborhood_epsilon = 0.1
t_step = 0.1
t = np.arange(0, entrainment_cycles*T ,t_step)


import ForcedHongModel as ForcedHongModel
mod_name = ForcedHongModel.mod_forced_hong_model
mod_name.c0 = copy(T)       # Zeitgeber period
mod_name.c1 = copy(kappa)   # Zeitgeber photo/thermoperiod
mod_name.c2 = copy(z0)      # Zeitgeber Amplitude
mod_name.c3 = copy(s)       # Zeitgeber steepness
mod_name.c4 = 0             # Phase shift of ZG in hours
mod_name.c22 = 2.           # Hill Coefficient


X0 = [1., 0] + state0
SetOscillatorParameters(rate)


def T_detection_epsilon_ball(solution_array, t_trans, neighborhood_epsilon, t_step):
    save_initial = solution_array[int(t_trans/t_step)].copy()
    distance_from_inital_state = sqrt(sum((solution_array - save_initial)**2, axis=1))
    points_inside_epsilon_ball_indices = argwhere( distance_from_inital_state <= neighborhood_epsilon ).flatten()
    leftmost_points_in_epsilon_ball = points_inside_epsilon_ball_indices[argwhere(points_inside_epsilon_ball_indices - roll(points_inside_epsilon_ball_indices, 1) != 1)].flatten()
    period_array = (leftmost_points_in_epsilon_ball[2:] - leftmost_points_in_epsilon_ball[1:-1])*t_step      # neglect 1st period
    return period_array

def numpy_simple_shooting(sol_vector, entrainment_cycles, t_trans_cycles, t_period, t_step, neighborhood_epsilon):
    save_initial = copy(sol_vector[int((t_trans_cycles)*t_period/t_step)])
    test_set =  sol_vector[array((t_trans_cycles + array(range(entrainment_cycles - t_trans_cycles)))*t_period/t_step, dtype=int)]
    c_x = argwhere(sqrt(sum((test_set - save_initial)**2, axis=1)) <= neighborhood_epsilon).flatten()
    return c_x[1:] - c_x[:-1]


T_2_mesh = [12.]*3 + [16.]*9 + [22.]*9 + [24.]*3 + [26.]*9
pp_2_mesh = [0.25, 0.5, 0.75] + [0.16, 0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.84] + [0.16, 0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.84] + [0.16, 0.25, 0.33] + [0.16, 0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.84]

EntrainmentResults_Exp_frqplus = array([2., 2., 2.] + [1., 1., 1., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [1.]*9 + [1.]*3 + [1.] + [1.]*7 + [np.nan])

paramlist =  zip(T_2_mesh,pp_2_mesh)

def CalculateOnionsF2py(params):
    T = params[0]
    kappa = params[1]
    mod_name.c0 = copy(T)
    mod_name.c1 = copy(kappa)
    try:
        t = arange(0, T*entrainment_cycles, t_step)
        c_hmax = (1.-(kappa))*T/2
        sol = odeint(mod_name.ode, X0, t, hmax=c_hmax)
        determine_per = T_detection_epsilon_ball(sol, t_trans_cycles, neighborhood_epsilon, t_step)
        c_a, c_b = mean(determine_per), std(determine_per)
        try:
            determine_per_shooting = numpy_simple_shooting(sol, entrainment_cycles, t_trans_cycles, T, t_step, neighborhood_epsilon)
            c_c, c_d = mean(determine_per_shooting), std(determine_per_shooting)
        except:
            c_c, c_d = 0., 0.
    except:
        c_a, c_b = 0., 0.
        print "Fail"
    return c_a, c_b, c_c, c_d


z_steps = 0.01
z_max = 1.5
z_array = arange(0, z_max, z_steps)

scaled_per = linspace(16., 32., len(z_array))

freerun_frqplus = 22.025
c_scale = freerun_frqplus / scaled_per
c_cutoff = 0.1
FitnessPlateau = zeros(len(z_array)*len(c_scale)).reshape(len(z_array),len(c_scale))
SavePath = "./ArnoldOnion/2DOptimizeOnion_scaledtau_1632_Zmax"+str(z_max)+"dz"+str(z_steps)+"dt"+str(t_step)+".p"
if os.path.exists(SavePath) != True:
    for c_n, c_scaling_rate in enumerate(c_scale):
        FitnessFunc = []
        for z0 in z_array:
            SetOscillatorParametersScalePeriod(rate, c_scaling_rate)
            mod_name.c2 = z0
            pool = multiprocessing.Pool()
            res  = pool.map(CalculateOnionsF2py,paramlist)
            tmp_mean = copy(array(res).T[2])
            tmp_fitness = sum(nan_to_num(tmp_mean, 0.) == nan_to_num(EntrainmentResults_Exp_frqplus, 0.))
            print
            print c_scaling_rate, z0
            print z0, tmp_fitness
            FitnessFunc.append(tmp_fitness)
            pool.close()    # important to clear memory
        FitnessPlateau[c_n] = FitnessFunc
    print FitnessPlateau
    SaveData = (z_array, c_scale, scaled_per, FitnessPlateau)
    pickle.dump(SaveData, open(SavePath, "w"))
else:
    LoadResults = pickle.load(open(SavePath, "r"))
    z_array, c_scale, scaled_per, FitnessPlateau = LoadResults
    print "Loaded"

imshow(FitnessPlateau, origin='lower', interpolation='nearest', extent=(min(z_array), max(z_array), min(c_scale), max(c_scale)), aspect="auto")

show()

