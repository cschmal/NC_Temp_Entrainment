"""
Code underlying Figure 5 A,B,C of the manuscript
'Principles underlying the complex dynamics of
temperature entrainment by a circadian clock'
"""

from pylab import*
import seaborn as sns
import pickle


t_step = 0.1
T_steps = 0.1
pp_steps = 0.005

T_min, T_max = 8., 28. +T_steps
pp_min, pp_max = 0, 1.+pp_steps


figure(figsize=(10.5, 3.5))
# loads and plots previously computed optimization results
subplot(132)
z_steps = 0.1
z_steps = 0.01
z_max = 1.5

# plot optimization results
SavePath = "./ArnoldOnion/2DOptimizeOnion_scaledtau_1632_Zmax"+str(z_max)+"dz"+str(z_steps)+"dt"+str(t_step)+".p"
LoadResults = pickle.load(open(SavePath, "r"))
z_array, c_scale, scaled_per, FitnessPlateau = LoadResults
scale_step = abs(scaled_per[1]- scaled_per[0])
imshow(FitnessPlateau.T, origin='lower', interpolation='nearest', extent=( min(scaled_per)-scale_step/2, max(scaled_per)+scale_step-scale_step/2, min(z_array) - z_steps/2, max(z_array)+z_steps-z_steps/2), aspect="auto", vmin=0, vmax=33)
xticks(range(12, 36+2, 2), fontsize=15)
xlim(18, 30)

z_array_coordinate, scale_array_coordinate = argmax(FitnessPlateau.flatten())%FitnessPlateau.shape[0], argmax(FitnessPlateau.flatten())/(FitnessPlateau.shape[0])      # gives the coordinates of the best hit
print max(FitnessPlateau.flatten())%FitnessPlateau.shape[0], z_array[z_array_coordinate], c_scale[scale_array_coordinate], scaled_per[scale_array_coordinate]


for c_num, c_max in enumerate(argwhere(FitnessPlateau.flatten() == max(FitnessPlateau.flatten())).flatten()):
    z_array_coordinate, scale_array_coordinate = c_max%FitnessPlateau.shape[0], c_max/(FitnessPlateau.shape[0])      # gives the coordinates of the best hit
    print c_num+1, c_max, z_array[z_array_coordinate], c_scale[scale_array_coordinate], scaled_per[scale_array_coordinate]
    plot( scaled_per[scale_array_coordinate], z_array[z_array_coordinate], marker="*", linestyle="", color="k" )
    
xticks([18, 20, 22, 24, 26, 28, 30], ["18", "", "22", "", "26", "", "30" ], fontsize=15)
yticks([0, 0.5, 1.], ["0", "0.5", "1" ], fontsize=15)
ylim(0-z_steps/2, 1)
xlabel(r"$\tau$ [h]" , fontsize=15)
ylabel("$z_0$ [a.u.]", fontsize=15)
title("B", loc="left", fontsize=15)
ax1 = gca()
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cbaxes = inset_axes(ax1, width="30%", height="3%", loc="upper center") 
cb = colorbar(cax=cbaxes, ticks=[0, 15, 33], orientation='horizontal')
cb.set_ticks([0, 16.5, 33])
cb.set_ticklabels(["0%", "", "100%"])
#cb = colorbar()
cb.set_label("Fitting Score")
    

z0, c_title, c_subplot = 0.1, "A", 131
subplot(c_subplot)
SavePath = "./ArnoldOnion/OnionZ"+str(z0)+"PPstep"+str(pp_steps)+"Tmin"+str(T_min)+"Tmax"+str(T_max)+"Tstep"+str(T_steps)+"dt"+str(t_step)+".npz"
LoadResults = np.load(SavePath, allow_pickle=True)["arr_0"]
T_2_mesh, pp_2_mesh, res = LoadResults
T_vec, pp_vec = meshgrid(T_2_mesh, pp_2_mesh)

print "Loaded"
current_palette = sns.color_palette() 

c_res_mean = array(res).T[2]
c_res_std = array(res).T[3]
c_cutoff = 0.1
c_res_mean[argwhere(c_res_std > c_cutoff)] = 0

Arnold1to1 = np.ma.masked_where(( c_res_mean != 1) , c_res_mean)
Arnold1to2 = np.ma.masked_where(( c_res_mean != 2) , c_res_mean+1)

c_snap = False
c_colormap = "tab10"
p1 = pcolor(T_vec, pp_vec, Arnold1to1.reshape(len(T_2_mesh),len(pp_2_mesh)).T, snap=c_snap, label="1:1", linewidth=0, vmin=1, vmax=11, cmap=c_colormap, rasterized=True)
p1.set_edgecolor('face')
p2 = pcolor(T_vec, pp_vec, Arnold1to2.reshape(len(T_2_mesh),len(pp_2_mesh)).T, snap=c_snap, label="1:2", linewidth=0, vmin=1, vmax=11, cmap=c_colormap, rasterized=True)
p2.set_edgecolor('face')

xticks(arange(10,28, 4), fontsize=15)
xlim(10, 28)
yticks([0, 0.5, 1.], fontsize=15)
ylim(0, 1)
xlabel("T [h]", fontsize=15)
ylabel(r"Thermoperiod $\varkappa_T$", fontsize=15)
title(c_title, loc="left", fontsize=15)

T_2_mesh = [12.]*3 + [16.]*9 + [22.]*9 + [24.]*3 + [26.]*9
pp_2_mesh = [0.25, 0.5, 0.75] + [0.16, 0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.84] + [0.16, 0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.84] + [0.16, 0.25, 0.33] + [0.16, 0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.84]
EntrainmentResults_Exp_frqplus = array([2., 2., 2.] + [1., 1., 1., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [1.]*9 + [1.]*3 + [1.] + [1.]*7 + [np.nan])
T_2_mesh  = array(T_2_mesh)
pp_2_mesh = array(pp_2_mesh)
scatter(T_2_mesh[argwhere(EntrainmentResults_Exp_frqplus==1.)].flatten(), pp_2_mesh[argwhere(EntrainmentResults_Exp_frqplus==1.)].flatten(), marker="o", color="gray")
scatter(T_2_mesh[argwhere(EntrainmentResults_Exp_frqplus==2.)].flatten(), pp_2_mesh[argwhere(EntrainmentResults_Exp_frqplus==2.)].flatten(), marker="^", color="gray")
scatter(T_2_mesh[argwhere(np.isnan(EntrainmentResults_Exp_frqplus))].flatten(), pp_2_mesh[argwhere(np.isnan(EntrainmentResults_Exp_frqplus))].flatten(), marker="x", color="gray")


c_title, c_subplot = "C", 133
subplot(c_subplot)

T_steps = 0.1
pp_steps = 0.005
T_min, T_max = 10., 30. +T_steps
pp_min, pp_max = 0+pp_steps, 1.+pp_steps
T_2_mesh, pp_2_mesh = arange(T_min, T_max, T_steps), arange(pp_min, pp_max, pp_steps)
T_vec, pp_vec = meshgrid(T_2_mesh, pp_2_mesh)
z0, c_scale = 0.47, 0.9495732060185184
SavePath = "./ArnoldOnion/PhasesScaledOnionZ"+str(z0)+"PPstep"+str(pp_steps)+"Tmin"+str(T_min)+"Tmax"+str(T_max)+"Tstep"+str(T_steps)+"dt"+str(t_step)+"Scale"+str(round(c_scale, 3))+".npz"


LoadResults = np.load(SavePath, allow_pickle=True)["arr_0"]
T_2_mesh, pp_2_mesh, res = LoadResults
print "Loaded"
current_palette = sns.color_palette() 

c_res_mean = array(res).T[2]
c_res_std = array(res).T[3]
c_cutoff = 0.1
c_res_mean[argwhere(c_res_std > c_cutoff)] = 0

Arnold1to1 = np.ma.masked_where(( c_res_mean != 1) , c_res_mean)
Arnold1to2 = np.ma.masked_where(( c_res_mean != 2) , c_res_mean+1)

c_snap = False
c_colormap = "tab10"
p1 = pcolor(T_vec, pp_vec, Arnold1to1.reshape(len(T_2_mesh),len(pp_2_mesh)).T, snap=c_snap, label="1:1", linewidth=0, vmin=1, vmax=11, cmap=c_colormap, rasterized=True)
p1.set_edgecolor('face')
p2 = pcolor(T_vec, pp_vec, Arnold1to2.reshape(len(T_2_mesh),len(pp_2_mesh)).T, snap=c_snap, label="1:2", linewidth=0, vmin=1, vmax=11, cmap=c_colormap, rasterized=True)
p2.set_edgecolor('face')

xticks(arange(10,28, 4), fontsize=15)
xlim(10, 28)
yticks([0, 0.5, 1.], fontsize=15)
ylim(0, 1)
xlabel("T [h]", fontsize=15)
ylabel(r"Thermoperiod $\varkappa_T$", fontsize=15)
title(c_title, loc="left", fontsize=15)


T_2_mesh = [12.]*3 + [16.]*9 + [22.]*9 + [24.]*3 + [26.]*9
pp_2_mesh = [0.25, 0.5, 0.75] + [0.16, 0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.84] + [0.16, 0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.84] + [0.16, 0.25, 0.33] + [0.16, 0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.84]
EntrainmentResults_Exp_frqplus = array([2., 2., 2.] + [1., 1., 1., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] + [1.]*9 + [1.]*3 + [1.] + [1.]*7 + [np.nan])

T_2_mesh  = array(T_2_mesh)
pp_2_mesh = array(pp_2_mesh)

scatter(T_2_mesh[argwhere(EntrainmentResults_Exp_frqplus==1.)].flatten(), pp_2_mesh[argwhere(EntrainmentResults_Exp_frqplus==1.)].flatten(), marker="o", color="gray")
scatter(T_2_mesh[argwhere(EntrainmentResults_Exp_frqplus==2.)].flatten(), pp_2_mesh[argwhere(EntrainmentResults_Exp_frqplus==2.)].flatten(), marker="^", color="gray")
scatter(T_2_mesh[argwhere(np.isnan(EntrainmentResults_Exp_frqplus))].flatten(), pp_2_mesh[argwhere(np.isnan(EntrainmentResults_Exp_frqplus))].flatten(), marker="x", color="gray")


tight_layout()


show()


    

