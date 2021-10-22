"""
Code underlying Figure 5 D,E,F of the manuscript
'Principles underlying the complex dynamics of
temperature entrainment by a circadian clock'
"""

from pylab import*
from scipy.integrate import odeint
from scipy.signal import argrelextrema
import seaborn as sns


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

# function to set parameter values of the clock model within the extension
# module "mod_name", taken from the parameter dictonary, including a scale 
# factor to scale the inrinsic period of the clock
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





# import extensiom module containing the Neurospora clock model
import ForcedHongModel as ForcedHongModel   
mod_name = ForcedHongModel.mod_forced_hong_model

# set simulation parameters
T       = 24.
kappa   = 0.5
s       = 100
entrainment_cycles = 520
t_trans_cycles = entrainment_cycles - 20
neighborhood_epsilon = 0.1
t_step = 0.1
t = np.arange(0, entrainment_cycles*T ,t_step)

mod_name.c0 = copy(T)            # Zeitgeber period
mod_name.c1 = copy(kappa)        # ZG thermoperiod
mod_name.c3 = copy(s)            # ZG steepness
mod_name.c4 = 0.                 # phase shift of ZG in [h]
mod_name.c22 = 2.                # Hill Coefficient


X0 = [1., 0] + state0
SetOscillatorParametersScalePeriod(rate, 1)

# Best hits from optimization protocol
z0, Scale = [0.55, 0.52, 0.48, 0.47], [0.9860952524038461, 0.9813770933014354, 0.9720749407582937, 0.9495732060185184]    

# choose one of the best hits
ParamSet = 3
c_Z     = z0[ParamSet]
c_scale = Scale[ParamSet] 

SetOscillatorParametersScalePeriod(rate, c_scale)

old_c5 = copy(mod_name.c5)
k1_min = old_c5*(1-copy(z0[ParamSet])/2)
k1_max = old_c5*(1+copy(z0[ParamSet])/2)


# compute bifurcation diagram
Z = 0.
mod_name.c0 = copy(T)
mod_name.c2 = copy(Z)
dk = 0.01
k1_array = arange(1., 2.5+dk, dk)
min_array = []
max_array = []
for k1 in k1_array:
    mod_name.c5 = k1
    sol = odeint(mod_name.ode, X0, t)
    x = sol.T[2][int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*24.) / t_step)]
    # for local maxima
    c_max = x[argrelextrema(x, np.greater)]
    # for local minima
    c_min = x[argrelextrema(x, np.less)]
    if len(c_min) == 0:
        min_array.append(x[-1])
        max_array.append(x[-1])
    else:
        min_array.append(mean(c_min))
        max_array.append(mean(c_max))

# plot bifurcation diagram into panel A
fig = figure(figsize=(10.5, 3.5))

subplot(131)

plot(k1_array, min_array, "k-")
plot(k1_array, max_array, "k-")
vlines(k1_min, 0, 10, linestyle="--")
vlines(old_c5, 0, 10, linestyle="--")
vlines(k1_max, 0, 10, linestyle="--")

text(old_c5, 9, "$k_1$", size=12, verticalalignment="center", horizontalalignment="center", bbox=dict(boxstyle="round", fc="white"))
text(k1_min, 9, r"$k_{{min}}$", size=12, verticalalignment="center", horizontalalignment="center", bbox=dict(boxstyle="round", fc="white"))
text(k1_max, 9, r"$k_{{max}}$", size=12, verticalalignment="center", horizontalalignment="center", bbox=dict(boxstyle="round", fc="white"))

xlabel("$frq$ transcriptional rate", fontsize=15)
ylabel("$frq$ mRNA amplitude", fontsize=15)
xticks([1., 1.5, 2, 2.5], ["1", "", "2.5", ""], fontsize=15)
xlim(1, 2.5)
#xlim(1., 3)
yticks([0, 5, 10], ["0", "", "10"], fontsize=15)
ylim(0, 10)

title("D", loc="left", fontsize=15)


# compute and plot simulation of free running dynamics
# for 
subplot(232)

Z = 0.      # no entrainment!
T = 22      # ZG period (not relevant)

mod_name.c0 = copy(T)
mod_name.c2 = copy(Z)

# choose k_min (i.e. the value the ZG drives k1 to for low temperatures) for k1
mod_name.c5 = copy(k1_min)

mod_name.c4 = (1.-kappa)*T/2+2.*T*kappa     # phase shift of ZG in [h], not relevant

# simulate and plot dynamics
t = arange(0, T*entrainment_cycles, t_step)
c_hmax = (1.-(kappa))*T/2
sol = odeint(mod_name.ode, X0, t)

plot(t, sol.T[2], label=r"$k_1=k_{{min}}$")
title("E", loc="left", fontsize=15)
yticks([0, 4, 8], ["0", "", "8"], fontsize=15)
ylim(0, 8)
xticks(arange(0*24, 7*24, 24), ["", "", "", "", "", "", ""], fontsize=15)
xlim(0, 24*6)
legend(loc=0)

# simulate and plot dynamics as above for k1=k_max
subplot(235)
mod_name.c5 = copy(k1_max)

sol = odeint(mod_name.ode, X0, t)
plot(t, sol.T[2], label=r"$k_1=k_{{max}}$")
xlabel("Time [d]", fontsize=15)
yticks([0, 4, 8], ["0", "", "8"], fontsize=15)
ylim(0, 8)
xticks(arange(0*24, 7*24, 24), ["0", "", "", "3", "", "", "6"], fontsize=15)
xlim(0, 24*6)
legend(loc="lower right")




# simulate entrained dynamics for T = 16 and kappa = 0.25
subplot(233)
Z     = z0[ParamSet]    # choose ZG strength from optimized parameter set
mod_name.c2 = copy(Z)
mod_name.c5 = old_c5    # choose nominal k1 value
kappa = 0.25            # choose ZG thermoperiod
T = 16.                 # choose ZG period
t = np.arange(0, entrainment_cycles*T ,t_step)
mod_name.c0 = copy(T)
mod_name.c1 = copy(kappa)
mod_name.c4 = (1.-kappa)*T/2+2.*T*kappa

#simulate dynamics
sol = odeint(mod_name.ode, X0, t)

# plot clock and ZG dynamics
plot(t, sol.T[2], label=r"$T$=$"+str(int(T))+r"h, \varkappa$="+str(kappa))
coeff=mod_name.c3*mod_name.c0**2/(2*pi*(mod_name.c1*mod_name.c0))
arg=sol.T[0]*cos(2.*pi*mod_name.c4/mod_name.c0)+sol.T[1]*sin(2.*pi*mod_name.c4/mod_name.c0)-cos(pi*mod_name.c1)
force=mod_name.c2*(arctan(coeff*arg)/pi)
fill_between(t[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*T) / t_step)], 0, 20*(Z/2-force)[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*T) / t_step)], color="lightgray")
xticks(arange(T*t_trans_cycles, T*t_trans_cycles+7.*T, T), ["", "", "", "", "", "", ""], fontsize=15)
xlim(T*t_trans_cycles, T*t_trans_cycles+6*T)
legend(loc="lower right")
yticks([0, 4, 8], ["0", "", "8"], fontsize=15)
ylim(0, 8)
title("F", loc="left", fontsize=15)


# simulate entrained dynamics as above for T = 24 and kappa = 0.25
subplot(236)
kappa = 0.25
T = 24.
t = np.arange(0, entrainment_cycles*T ,t_step)
mod_name.c0 = copy(T)
mod_name.c1 = copy(kappa)
mod_name.c4 = (1.-kappa)*T/2+2.*T*kappa

sol = odeint(mod_name.ode, X0, t)

plot(t, sol.T[2], label=r"$T$=$"+str(int(T))+r"h, \varkappa$="+str(kappa))
coeff=mod_name.c3*mod_name.c0**2/(2*pi*(mod_name.c1*mod_name.c0))
arg=sol.T[0]*cos(2.*pi*mod_name.c4/mod_name.c0)+sol.T[1]*sin(2.*pi*mod_name.c4/mod_name.c0)-cos(pi*mod_name.c1)
force=mod_name.c2*(arctan(coeff*arg)/pi)
fill_between(t[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*T) / t_step)], 0, 20*(Z/2-force)[int(T*t_trans_cycles / t_step):int((T*t_trans_cycles+8*T) / t_step)], color="lightgray")
xticks(arange(T*t_trans_cycles, T*t_trans_cycles+7.*T, T), ["0", "", "", "3", "", "", "6"], fontsize=15)
xlim(T*t_trans_cycles, T*t_trans_cycles+6*T)
legend(loc="lower right")
yticks([0, 4, 8], ["0", "", "8"], fontsize=15)
ylim(0, 8)
xlabel("Time [d]", fontsize=15)


tight_layout()


show()




