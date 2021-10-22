"""
Code underlying Figure 6 of the manuscript
'Principles underlying the complex dynamics of
temperature entrainment by a circadian clock'
"""

from pylab import*
import matplotlib.collections as collections
import seaborn as sns
import pickle
import os

c_color_palette = [ "tab:blue", "tab:green", "tab:purple" , "tab:orange"]

figure(figsize=(15, 3.5))

# plot mean and standard deviation of conidiation patterns for T=22 and T=26 at a thermoperiod of 0.5
subplot(143)
for c_i, c_T in enumerate([22, 26]):
    t, v1, v2, v3 = loadtxt("./Data/WT_T"+str(c_T)+"_TP50.txt", unpack=True, usecols=(0,1,2,3))

    T = c_T
    warm = 50
    t = t/60.   # time in h
    dummy = 1-(warm/100.)
    cold_dur = dummy*T
    MeanInt  = mean(array([v1, v2, v3]), axis=0)
    MeanMStd = MeanInt - std(array([v1, v2, v3]), axis=0)
    MeanPStd = MeanInt + std(array([v1, v2, v3]), axis=0)
    dt = mean(array(t[1:])-array(t[:-1]))
    starting_index = argwhere(t >= T+cold_dur).flatten()[0]
    t_tmp = t[starting_index:]
    t_tmp -= t_tmp[0]
    t_tmp /= c_T
    plot(t_tmp, MeanInt[starting_index:], color=c_color_palette[c_i], label="T="+str(c_T)+"h")
    fill_between(t_tmp, MeanMStd[starting_index:], MeanPStd[starting_index:], color=c_color_palette[c_i] , alpha=0.4)

collection = collections.BrokenBarHCollection.span_where(t_tmp, ymin=700, ymax=-700, where= ((t_tmp % 1) >= (T-cold_dur)/T), facecolor="gray", alpha=0.5)
ax = plt.gca()
ax.add_collection(collection)

title("C", fontsize=20, loc="left")
ylim(-300, 500)
xticks(range(0, 8), ["0", "1T", "2T", "3T", "4T", "5T", "6T", "7T"], fontsize=15)
xlim(0, 7)
xlim(3, 5.5)
yticks([-300, -200, -100, 0, 100, 200, 300, 400, 500], ["-300", "", "", "0", "","", "300", "", ""], fontsize=15)
xlabel("Time", fontsize=15)
ylabel("Conidiation Intensity [a.u.]", fontsize=15)
legend(loc="lower right")

# plot mean and standard deviation of conidiation patterns for T=22 and different thermoperiods of 0.25, 0.5, 0.75
subplot(144)
c_T = 22
for c_i, c_pp in enumerate([25, 50, 75]):
    t, v1, v2, v3 = loadtxt("./Data/WT_T"+str(c_T)+"_TP"+str(c_pp)+".txt", unpack=True, usecols=(0,1,2,3))

    T = c_T
    warm = c_pp
    t = t/60.       # time in h
    dummy = 1-(warm/100.)
    cold_dur = dummy*T
    MeanInt  = mean(array([v1, v2, v3]), axis=0)
    MeanMStd = MeanInt - std(array([v1, v2, v3]), axis=0)
    MeanPStd = MeanInt + std(array([v1, v2, v3]), axis=0)
    dt = mean(array(t[1:])-array(t[:-1]))
    starting_index = argwhere(t >= T+cold_dur).flatten()[0]
    t_tmp = t[starting_index:]
    t_tmp -= t_tmp[0]
    t_tmp /= c_T
    plot(t_tmp, MeanInt[starting_index:], color=c_color_palette[c_i], label=r"$\varkappa_T$="+str(c_pp)+"%")
    fill_between(t_tmp, MeanMStd[starting_index:], MeanPStd[starting_index:], color=c_color_palette[c_i] , alpha=0.4)

    collection = collections.BrokenBarHCollection.span_where(t_tmp, ymin=500-50*c_i, ymax=450-50*c_i, where= ((t_tmp % 1) >= (T-cold_dur)/T), facecolor=c_color_palette[c_i], alpha=0.5)
    ax = plt.gca()
    ax.add_collection(collection)

title("D", fontsize=20, loc="left")
text(5.9, -250, "$T=22$h", fontsize=10, bbox=dict(boxstyle="round", facecolor='w', edgecolor="gray", alpha=0.8))
ylim(-300, 500)
xticks(range(0, 8), ["0", "1T", "2T", "3T", "4T", "5T", "6T", "7T"], fontsize=15)
xlim(0, 7)
xlim(3, 5.5)
yticks([-300, -200, -100, 0, 100, 200, 300, 400, 500], ["-300", "", "", "0", "","", "300", "", ""], fontsize=15)
xlabel("Time", fontsize=15)
ylabel("Conidiation Intensity [a.u.]", fontsize=15)
legend(loc="lower right")

# plot cross section through previously calculated Arnold onion at T=22 and T=26
t_step = 0.1
T_steps = 0.1
pp_steps = 0.005
T_min, T_max = 10., 30. +T_steps
pp_min, pp_max = 0+pp_steps, 1.+pp_steps
T_2_mesh, pp_2_mesh = arange(T_min, T_max, T_steps), arange(pp_min, pp_max, pp_steps)
T_vec, pp_vec = meshgrid(T_2_mesh, pp_2_mesh)
z0, c_scale = 0.47, 0.9495732060185184

# load simulated Arnold onion data
SavePath = "./ArnoldOnion/PhasesScaledOnionZ0.47PPstep0.005Tmin10.0Tmax30.1Tstep0.1dt0.1Scale0.95.npz"

LoadResults = np.load(SavePath, allow_pickle=True)["arr_0"]
T_2_mesh, pp_2_mesh, res = LoadResults

c_res_mean = array(res).T[2]
c_res_std = array(res).T[3]
c_cutoff = 0.1
c_res_mean[argwhere(c_res_std > c_cutoff)] = 0
amp_threshold = 0.00001
tmp_phase = np.ma.masked_where((c_res_mean != 1) ,array(res).T[6])
tmp_phase = np.ma.array(tmp_phase, mask=(array(res).T[9] >= amp_threshold))


# plot cross sections at T=22 and T=26
subplot(141)
c_linewidth = 3
c=0
for target_per in [22, 26]:
    target_ind = int((target_per - T_min)/T_steps)
    scatter(pp_2_mesh, (360*tmp_phase.reshape(len(T_2_mesh),len(pp_2_mesh)).T/T_vec).T[target_ind], label="T="+str(target_per)+"h", alpha=0.6, s=10, marker="o", color=c_color_palette[c])
    c+=1
xticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=15)
xlim(0, 1)
yticks([0, 180, 360], ["0$^{\circ}$", "180$^{\circ}$", "360$^{\circ}$"], fontsize=15)
ylim(0, 360)
xlabel(r"Thermoperiod $\varkappa_T$", fontsize=15)
ylabel(r"FRQ$_c$ Phase", fontsize=15)
legend(loc=0)
title("A", loc="left", fontsize=20)


# plot mean phases and standard deviation from peak picking for T=22 and T=26 at different thermoperiods
# last point for T=26 has been omitted as the corresponding dynamics have been classified as non-entrained,
# i.e. a stable phase of entrainment is not well defined

subplot(142)

c_T = 22
PPArray, PhaseMeanArray, PhaseStdArray = pickle.load(open("./Data/PeackPickingPhases_T"+str(c_T)+".p", 'rb'))
errorbar(array(PPArray, dtype=float)/100, array(PhaseMeanArray)/2/pi*360, yerr=array(PhaseStdArray)/2/pi*360,  label="T="+str(c_T)+"h", linewidth=3, marker="o", alpha=0.8, color=c_color_palette[0])

c_T = 26
PPArray, PhaseMeanArray, PhaseStdArray = pickle.load(open("./Data/PeackPickingPhases_T"+str(c_T)+".p", 'rb'))
errorbar(array(PPArray[:-1], dtype=float)/100, array(PhaseMeanArray[:-1])/2/pi*360, yerr=array(PhaseStdArray[:-1])/2/pi*360,  label="T="+str(c_T)+"h", linewidth=3, marker="o", alpha=0.8, color=c_color_palette[1])

xticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=15)
xlim(0,1)
xlabel(r"Thermoperiod $\varkappa_T$", fontsize=15)
yticks([-180, 0, 180], ["180$^{\circ}$", "0$^{\circ}$", "180$^{\circ}$"], fontsize=15)
ylim(-180, 180)
ylabel(r"Conidiation Phase", fontsize=15)
legend(loc=0)
title("B", loc="left", fontsize=20)


tight_layout()


show()
