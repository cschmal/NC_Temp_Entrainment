"""
Code underlying Figure 4 of the manuscript
'Principles underlying the complex dynamics of
temperature entrainment by a circadian clock'
"""

from pylab import*
from scipy.integrate import odeint
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import seaborn as sns
import pickle

current_palette = sns.color_palette() 

figure(figsize=(10.5, 3.5*2))


# Plot Arnold tongue for a given set of simulation parameters
kappa = 0.75                    # thermoperiod
T_steps = 0.1                   # ZG perdiod sampling interval
Z_steps = 0.005                 # ZG amplitude sampling interval
T_min, T_max = 8., 32. +T_steps # range of ZG period values
Z_min, Z_max = 0, 1.+Z_steps    # range of ZG amplitude valuess
T_2_mesh, Z_2_mesh = arange(T_min, T_max, T_steps), arange(Z_min, Z_max, Z_steps)
T_vec, Z_vec = meshgrid(T_2_mesh, Z_2_mesh)
# Arnold tongue has been calculated and stored previously, e.g. using the code from "Figure3.py"
SavePath = "./ArnoldOnion/Arnold_PP"+str(kappa)+"_Zmin"+str(Z_min)+"Zmax"+str(Z_max)+"Zstep"+str(Z_steps)+"Tmin"+str(T_min)+"Tmax"+str(T_max)+"Tstep"+str(T_steps)+".npz"
LoadResults = np.load(SavePath, allow_pickle=True)["arr_0"]
T_2_mesh, Z_2_mesh, res = LoadResults
c_res_mean = array(res).T[2]
c_res_std = array(res).T[3]
c_res_mean[argwhere(c_res_std > 0.1)] = 0
c_cutoff = 0.1
# select 1:1 sync region
Arnold1to1 = np.ma.masked_where(( c_res_mean != 1) , c_res_mean)

# plot Arnold tongue in panel A
subplot(231)
c_snap = False
c_colormap = "tab10"
p1 = pcolor(T_vec, Z_vec, Arnold1to1.reshape(len(T_2_mesh),len(Z_2_mesh)).T, snap=c_snap, label="1:1", linewidth=0, vmin=1, vmax=11, cmap=c_colormap, rasterized=True)
p1.set_edgecolor('face')
xticks(arange(16,30, 4), fontsize=15)
xlim(15, 29)
yticks([0, 0.4, 0.8], fontsize=15)
ylim(0, 0.8)
xlabel("Zeitgeber Period T [h]", fontsize=15)
ylabel("Zeitgeber Strength $z_0$ [a.u.]", fontsize=15)
title("A", loc="left", fontsize=20)

# plot arrows pointing to parameter values, tested in exemplary simulations
annotate('B', xy=(21.37, 0.04), xytext=(16, 0.04),
            arrowprops=dict(facecolor='black', shrink=0.05),
            verticalalignment='center',
            fontsize=15
            )

annotate('C', xy=(19.58, 0.2), xytext=(16, 0.2),
            arrowprops=dict(facecolor='black', shrink=0.05),
            verticalalignment='center',
            fontsize=15
            )


annotate('E', xy=(27.9, 0.3), xytext=(25, 0.3),
            arrowprops=dict(facecolor='black', shrink=0.05),
            verticalalignment='center',
            fontsize=15
            )

annotate('F', xy=(27.9, 0.54), xytext=(25, 0.54),
            arrowprops=dict(facecolor='black', shrink=0.05),
            verticalalignment='center',
            fontsize=15
            )

# plot vertical line at period value T for which strobiscopic map has been computed
vlines(27.9, 0, 0.8, linestyle="--", linewidth=2, color="k")

# load and plot data of stroboscopic map for a given set of dZ and T
subplot(234)
dZ, T = 0.001, 27.9
Zarray, bifarray = pickle.load(open("./Data/BifDataT"+str(T)+"_dZ"+str(dZ)+".p", "rb"))
for Z, S in zip(Zarray, bifarray):
    scatter([Z]*len(S), S, marker="o", color="k", alpha=.2, s=0.05)
vlines(0.3, 7, 35, linestyle="--", linewidth=2, color="gray")
vlines(0.54, 7, 35, linestyle="--", linewidth=2, color="gray")
xlim(0, 0.8)
yticks([10, 20, 30], fontsize=15)
ylim(ymin=7, ymax=35)
xlabel("Zeitgeber strength $z_0$", fontsize=15)
ylabel(r"FRQc Stroboscopic Map", fontsize=15)
title("D", loc="left", fontsize=20)


# simulate and plot time series data for example parameter values as indicated by the arrows in panel A
c_plot1, c_plot2, c_plot3 = 7, 3, 2     # WC-1n, FRQc, frq mRNA

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

#==============================================================================
# initial conditions
#==============================================================================
frq_mrna0    = 4.5
frq_c0       = 25
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


s=100
import ForcedHongModel as ForcedHongModel
mod_name = ForcedHongModel.mod_forced_hong_model
mod_name.c3 = copy(s)                               # zeitgeber steepness
mod_name.c4 = 0                                     # phase shift of ZG in [h]
mod_name.c22 = 2.                                   # Hill coefficient

# parameter values and initial conditions
kappa = 0.75        # thermoperiod
X0 = [1., 0] + state0

# loop over different values of T and z0 and simulate time series and corresponding stroboscopic map data
for T, c_Z, c_subplot, c_title, c_final in zip([21.37, 19.58, 27.9, 27.9], [0.04, 0.2, 0.3, 0.54], [232, 233, 235, 236], ["B", "C", "E", "F"], [700, 50, 20, 50]):
    entrainment_cycles = 5500
    t_trans_cycles = entrainment_cycles - 20
    neighborhood_epsilon = 0.1
    t_step = 0.01
    t = np.arange(0, entrainment_cycles*T, t_step)
    transient_points = int(5250*T/t_step) + int(round((0.1*T)/t_step))
    transient_points2 = int((entrainment_cycles-c_final)*T/t_step)
    strobo_points = int(round(T/t_step))
    plot_cycles = 10
    SetOscillatorParameters(rate)
    Z = c_Z
    mod_name.c0 = copy(T)
    mod_name.c1 = copy(kappa)                   # ZG thermoperiod
    mod_name.c2 = copy(Z)                       # ZG amplitude
    mod_name.c4 = (1.-kappa)*T/2+2.*T*kappa     # Phase shift of ZG in [h]
    c_hmax = (1.-(kappa))*T/2
    sol = odeint(mod_name.ode, X0, t)

    subplot(c_subplot)
    plot(sol.T[c_plot3][transient_points:], sol.T[c_plot2][transient_points:], alpha=1., color="k", linewidth=0.05)
    # plot stroboscopic points
    scatter(sol.T[c_plot3][transient_points:][::strobo_points], sol.T[c_plot2][transient_points:][::strobo_points], marker=".", alpha=None, color=current_palette[1], s=15)

    xticks([0, 5, 10], fontsize=15)
    xlim(0, 10)
    yticks([10, 20, 30, 40], fontsize=15)
    ylim(9, 43)
    xlabel("FRQ mRNA", fontsize=15)
    ylabel(r"FRQc", fontsize=15)
    title(c_title, loc="left", fontsize=20)

    ax = gca()
    sax = inset_axes(ax, 
                        width="30%", # width = 30% of parent_bbox
                        height="30%", # height : 1 inch
                        loc="lower right")
    plot(t[transient_points2:], sol.T[c_plot2][transient_points2:], color="k", linewidth=0.2)
    xlim(t[transient_points2:][0], t[transient_points2:][-1])
    xticks([t[transient_points2:][0], t[transient_points2:][int(round(len(t[transient_points2:])/2))], t[transient_points2:][-1]], ["0T", "", str(c_final)+"T"], fontsize=7, ha="right")
    sax.xaxis.tick_top()
    yticks([10, 20, 30, 40], ["", "", "", ""], fontsize=15)
    ylim(9, 43)
    ylabel(r"FRQc", fontsize=10)


tight_layout()


show()
