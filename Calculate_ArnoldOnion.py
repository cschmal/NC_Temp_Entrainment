"""
Code underlying the calculation of Arnold onions 
within the manuscript
'Principles underlying the complex dynamics of
temperature entrainment by a circadian clock'
"""

from pylab import*
from scipy.integrate import odeint
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import multiprocessing
import os

#==============================================================================
# parameter dictionaries
#==============================================================================
# rate constants per hour for the wt clock model
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

def find_local_extrema_numpy_filtered(ga, sp, ep):
    """
    ga = given array
    sp = starting point of search in array
    ep = end point of search in array
    """
    i_min = where((ga[sp:ep][1:-1] < ga[sp:ep][2:]) *  (ga[sp:ep][1:-1] < ga[sp:ep][:-2]))[0] + 1
    i_max = where((ga[sp:ep][1:-1] > ga[sp:ep][2:]) *  (ga[sp:ep][1:-1] > ga[sp:ep][:-2]))[0] + 1
    a_min = ga[sp:ep][i_min]
    a_max = ga[sp:ep][i_max]
    i_max = i_max[where(a_max > 0.99 * (max(a_max) - min(a_min)) + min(a_min))]
    a_max = ga[sp:ep][i_max]
    result_max = append([i_max], [a_max], axis=0)
    result_min = append([i_min], [a_min], axis=0)
    result = result_max, result_min
    return result


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
z0= 0.5

entrainment_cycles = 520
t_trans_cycles = entrainment_cycles - 20
neighborhood_epsilon = 0.1
t_step = 0.1
t = np.arange(0, entrainment_cycles*T ,t_step)


import ForcedHongModel as ForcedHongModel
mod_name = ForcedHongModel.mod_forced_hong_model

mod_name.c0 = copy(T)                       # ZG period
mod_name.c1 = copy(kappa)                   # ZG thermoperiod
mod_name.c2 = copy(z0)                      # ZG amplitude
mod_name.c3 = copy(s)                       # ZG steepness
mod_name.c4 = mod_name.c0*mod_name.c1/2.
mod_name.c22 = 2.                           # Hill Coefficient

# set initial conditions and model parameters
X0 = [1., 0] + state0
SetOscillatorParameters(rate)

T_steps = 0.1
pp_steps = 0.005

T_min, T_max = 10., 30. +T_steps
pp_min, pp_max = 0+pp_steps, 1.+pp_steps
T_2_mesh, pp_2_mesh = arange(T_min, T_max, T_steps), arange(pp_min, pp_max, pp_steps)
T_vec, pp_vec = meshgrid(T_2_mesh, pp_2_mesh)

paramlist = list(itertools.product(T_2_mesh,pp_2_mesh))

def CalculateOnionsF2py(params):
    T = params[0]
    kappa = params[1]
    mod_name.c0 = copy(T)
    mod_name.c1 = copy(kappa)
    mod_name.c4 = mod_name.c0*mod_name.c1/2.
    try:
        t = arange(0, T*entrainment_cycles, t_step)
        c_hmax = (1.-(kappa))*T/2
        sol = odeint(mod_name.ode, X0, t, hmax=c_hmax)
        determine_per = T_detection_epsilon_ball(sol, t_trans_cycles, neighborhood_epsilon, t_step)
        c_a, c_b = mean(determine_per), std(determine_per)
        determine_per_alt_var1 = find_local_extrema_numpy_filtered(sol.T[3], int((t_trans_cycles)*T/t_step), len(sol.T[3]))
        c_per1 = (determine_per_alt_var1[0][0][1:]-determine_per_alt_var1[0][0][:-1])*t_step
        c_e = mean(c_per1)
        c_f = std(c_per1)
        phases_1 = ((determine_per_alt_var1[0][0]+int((t_trans_cycles)*T/t_step))*t_step)%T
        c_g = mean(phases_1)
        c_h = std(phases_1)
        c_i = mean(determine_per_alt_var1[0][1]) - mean(determine_per_alt_var1[1][1])
        c_j = std(determine_per_alt_var1[0][1])
        c_k = std(determine_per_alt_var1[1][1])
        try:
            determine_per_shooting = numpy_simple_shooting(sol, entrainment_cycles, t_trans_cycles, T, t_step, neighborhood_epsilon)
            c_c, c_d = mean(determine_per_shooting), std(determine_per_shooting)
        except:
            c_c, c_d = 0., 0.
    except:
        c_a, c_b, c_c, c_d, c_e, c_f, c_g, c_h, c_i, c_j, c_k = 0., 0., 0., 0., 0., 0. ,0., 0., 0., 0., 0.
        print "Fail"
    return c_a, c_b, c_c, c_d, c_e, c_f, c_g, c_h, c_i, c_j, c_k



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


for z0, c_scale in zip([0.55, 0.52, 0.48, 0.47], [0.9860952524038461, 0.9813770933014354, 0.9720749407582937, 0.9495732060185184]):    # Best hits from Optimization
    SetOscillatorParametersScalePeriod(rate, c_scale)
    mod_name.c2 = copy(z0)      # Zeitgeber Amplitude
    print mod_name.c4
    SavePath = "./ArnoldOnion/PhasesScaledOnionZ"+str(z0)+"PPstep"+str(pp_steps)+"Tmin"+str(T_min)+"Tmax"+str(T_max)+"Tstep"+str(T_steps)+"dt"+str(t_step)+"Scale"+str(round(c_scale, 3))+".npz"
    if os.path.exists(SavePath) != True:
        print "Doing Calculation"
        pool = multiprocessing.Pool()
        res  = pool.map(CalculateOnionsF2py,paramlist)
        SaveData = (T_2_mesh, pp_2_mesh, res)
        np.savez_compressed(SavePath, SaveData)
    else:
        LoadResults = np.load(SavePath)["arr_0"]
        T_2_mesh, pp_2_mesh, res = LoadResults
        print "Loaded"
