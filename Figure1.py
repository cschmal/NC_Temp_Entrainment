"""
Code underlying Figure 1 of the manuscript
'Principles underlying the complex dynamics of
temperature entrainment by a circadian clock'

For three exemplary conditions (3 replicates eacg)
we investigate the raw time series, instantaneous phases
and Lomb-Scargle periodograms.
"""

from pylab import*
from scipy.signal import hilbert
from astropy.stats import LombScargle
import numpy as np
import matplotlib.collections as collections
import seaborn as sns
import os

# Set up plotting options
sns.set(style = "ticks")
c_color_palette = [ "tab:blue", "tab:green", "tab:purple" ]

# Load firt three time series
t, v1, v2, v3 = loadtxt("./Data/WT_T22_TP50.txt", unpack=True, usecols=(0,1,2,3))
T = 12                  # Zeitheber (ZG) period
warm = 50               # thermoperiod        
t = t/60.               # time in [h]     
dummy = 1-(warm/100.)
cold_dur = dummy*T      # cold phase of ZG


figure(figsize=(15, 10.5))

# Panel A
subplot(3, 3, 1)

# Plot raw time series
plot(t, v1, c_color_palette[0], linewidth=2)
plot(t, v2, c_color_palette[1], linewidth=2)
plot(t, v3, c_color_palette[2], linewidth=2)
collection = collections.BrokenBarHCollection.span_where(t, ymin=700, ymax=-700, where= ((t % T) <= cold_dur), facecolor="gray", alpha=0.5)
ax = plt.gca()
ax.add_collection(collection)
text(5, 350, "frq$^+$, $T=22$h", fontsize=15, bbox=dict(boxstyle="round", facecolor='w', edgecolor="k", alpha=0.8))
xlim(0, 176)
ylim(-300, 450)
xticks(range(0, 176, 24), fontsize=15)
yticks(fontsize=15)
xlabel("Time [h]", fontsize=15)
ylabel("Conidiation Intensity [a.u.]", fontsize=15)
title("A", fontsize=20, loc="left")


# Panel B
subplot(3, 3, 4)

# Mean detrending
v1=v1-mean(v1)
v2=v2-mean(v2)
v3=v3-mean(v3)

# Hilbert transformation
hv1 = hilbert(v1)
hv2 = hilbert(v2)
hv3 = hilbert(v3)

# Ignore first 2 days as transient dynamics
hv1 = hv1[argwhere(t>=48)].flatten()
hv2 = hv2[argwhere(t>=48)].flatten()
hv3 = hv3[argwhere(t>=48)].flatten()
t = t[argwhere(t>=48)].flatten()

# Plot instantaneous conidiation vs zeitgeber phase
scatter((2.*pi/T * (t-cold_dur-T/2) )%(2.*pi)-pi, angle(hv1), alpha=0.5, edgecolors=None, s=10, color=c_color_palette[0])
scatter((2.*pi/T * (t-cold_dur-T/2) )%(2.*pi)-pi, angle(hv2), alpha=0.5, edgecolors=None, s=10, color=c_color_palette[1])
scatter((2.*pi/T * (t-cold_dur-T/2) )%(2.*pi)-pi, angle(hv3), alpha=0.5, edgecolors=None, s=10, color=c_color_palette[2])
xlim(-pi, pi)
ylim(-pi, pi)
xlabel("Zeitgeber Phase [rad]", fontsize=15)
ylabel("Conidiation Phase [rad]", fontsize=15)
xticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"], fontsize=15)
yticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"], fontsize=15)
title("B", fontsize=20, loc="left")


# Panel C
subplot(3, 3, 7)

# Ignore first 2 days as transient dynamics
v1 = v1[argwhere(t>=48)].flatten()
v2 = v2[argwhere(t>=48)].flatten()
v3 = v3[argwhere(t>=48)].flatten()

# Define periods to be tested by LS-periodogram
TLS = arange(1., 60, 0.01)

# Lomb-Scargle analysis
ls1 = LombScargle(t, v1)
power_givenW1    = ls1.power(1./TLS)
ls2 = LombScargle(t, v2)
power_givenW2    = ls2.power(1./TLS)
ls3 = LombScargle(t, v3)
power_givenW3    = ls3.power(1./TLS)

# Plot Lomb-Scargle periodogram
plot(TLS, power_givenW1, color=c_color_palette[0])
plot(TLS, power_givenW2, color=c_color_palette[1])
plot(TLS, power_givenW3, color=c_color_palette[2])
xlabel("Period [h]", fontsize=15)
ylabel("Power Spectral Density", fontsize=15)
vlines(T, 0, max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()) + 0.1*max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()), color="gray", linestyle="--" )
vlines(2.*T, 0, max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()) + 0.1*max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()), color="gray", linestyle=":" )
xticks(arange(0, TLS[-1]+24, 24), fontsize=15)
xlim(0, TLS[-1])
ylim(0, max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()) + 0.1*max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()))
title("C", fontsize=20, loc="left")


# Since panels D-I are computed identically (only imported data differs), following code is no further commented.

subplot(3, 3, 2)

t, v1, v2, v3 = loadtxt("./Data/WT_T16_TP50.txt", unpack=True, usecols=(0,1,2,3))

T = 16
warm =50
t = t/60. #in h
dummy = 1-(warm/100.)
cold_dur = dummy*T

plot(t, v1, c_color_palette[0], linewidth=2)
plot(t, v2, c_color_palette[1], linewidth=2)
plot(t, v3, c_color_palette[2], linewidth=2)
collection = collections.BrokenBarHCollection.span_where(t, ymin=700, ymax=-700, where= ((t % T) <= cold_dur), facecolor="gray", alpha=0.5)
ax = plt.gca()
ax.add_collection(collection)
text(5, 350, "frq$^+$, $T=16$h", fontsize=15, bbox=dict(boxstyle="round", facecolor='w', edgecolor="k", alpha=0.8))
xlim(0, 176)
ylim(-300, 450)
xlabel("Time [h]", fontsize=15)
xticks(range(0, 176, 24), fontsize=15)
yticks(fontsize=15)
title("D", fontsize=20, loc="left")


subplot(3, 3, 5)

v1=v1-mean(v1)
v2=v2-mean(v2)
v3=v3-mean(v3)

hv1 = hilbert(v1)
hv2 = hilbert(v2)
hv3 = hilbert(v3)

hv1 = hv1[argwhere(t>=48)].flatten()
hv2 = hv2[argwhere(t>=48)].flatten()
hv3 = hv3[argwhere(t>=48)].flatten()
t = t[argwhere(t>=48)].flatten()

scatter((2.*pi/T * (t-cold_dur-T/2) )%(2.*pi)-pi, angle(hv1), alpha=0.5, edgecolors=None, s=10, color=c_color_palette[0])
scatter((2.*pi/T * (t-cold_dur-T/2) )%(2.*pi)-pi, angle(hv2), alpha=0.5, edgecolors=None, s=10, color=c_color_palette[1])
scatter((2.*pi/T * (t-cold_dur-T/2) )%(2.*pi)-pi, angle(hv3), alpha=0.5, edgecolors=None, s=10, color=c_color_palette[2])
xlim(-pi, pi)
ylim(-pi, pi)
xlabel("Zeitgeber Phase [rad]", fontsize=15)
xticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"], fontsize=15)
yticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"], fontsize=15)
title("E", fontsize=20, loc="left")


subplot(3, 3, 8)

v1 = v1[argwhere(t>=48)].flatten()
v2 = v2[argwhere(t>=48)].flatten()
v3 = v3[argwhere(t>=48)].flatten()

TLS = arange(1., 60, 0.01)

ls1 = LombScargle(t, v1)
power_givenW1    = ls1.power(1./TLS)
ls2 = LombScargle(t, v2)
power_givenW2    = ls2.power(1./TLS)
ls3 = LombScargle(t, v3)
power_givenW3    = ls3.power(1./TLS)

plot(TLS, power_givenW1, color=c_color_palette[0])
plot(TLS, power_givenW2, color=c_color_palette[1])
plot(TLS, power_givenW3, color=c_color_palette[2])
xlabel("Period [h]", fontsize=15)
ylabel("Power Spectral Density", fontsize=15)
vlines(T, 0, max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()) + 0.1*max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()), color="gray", linestyle="--" )
vlines(2.*T, 0, max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()) + 0.1*max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()), color="gray", linestyle=":" )
xticks(arange(0, TLS[-1]+24, 24), fontsize=15)
xlim(0, TLS[-1])
ylim(0, max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()) + 0.1*max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()))
title("F", fontsize=20, loc="left")


subplot(3, 3, 3)

t, v1, v2, v3 = loadtxt("./Data/WT_T12_TP50.txt", unpack=True, usecols=(0,1,2,3))

T = 12
warm =50
t = t/60. #in h
dummy = 1-(warm/100.)
cold_dur = dummy*T

plot(t, v1, c_color_palette[0], linewidth=2)
plot(t, v2, c_color_palette[1], linewidth=2)
plot(t, v3, c_color_palette[2], linewidth=2)
collection = collections.BrokenBarHCollection.span_where(t, ymin=700, ymax=-700, where= ((t % T) <= cold_dur), facecolor="gray", alpha=0.5)
ax = plt.gca()
ax.add_collection(collection)
text(5, 350, "frq$^+$, $T=12$h", fontsize=15, bbox=dict(boxstyle="round", facecolor='w', edgecolor="k", alpha=0.8))
xlim(0, 176)
ylim(-300, 450)
xlabel("Time [h]", fontsize=15)
xticks(range(0, 176, 24), fontsize=15)
yticks(fontsize=15)
title("G", fontsize=20, loc="left")


subplot(3, 3, 6)

v1=v1-mean(v1)
v2=v2-mean(v2)
v3=v3-mean(v3)

hv1 = hilbert(v1)
hv2 = hilbert(v2)
hv3 = hilbert(v3)

hv1 = hv1[argwhere(t>=48)].flatten()
hv2 = hv2[argwhere(t>=48)].flatten()
hv3 = hv3[argwhere(t>=48)].flatten()
t = t[argwhere(t>=48)].flatten()

scatter((2.*pi/T * (t-cold_dur-T/2) )%(2.*pi)-pi, angle(hv1), alpha=0.5, edgecolors=None, s=10, color=c_color_palette[0])
scatter((2.*pi/T * (t-cold_dur-T/2) )%(2.*pi)-pi, angle(hv2), alpha=0.5, edgecolors=None, s=10, color=c_color_palette[1])
scatter((2.*pi/T * (t-cold_dur-T/2) )%(2.*pi)-pi, angle(hv3), alpha=0.5, edgecolors=None, s=10, color=c_color_palette[2])
xlim(-pi, pi)
ylim(-pi, pi)
xlabel("Zeitgeber Phase [rad]", fontsize=15)
xticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"], fontsize=15)
yticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"], fontsize=15)
title("H", fontsize=20, loc="left")

subplot(3, 3, 9)

v1 = v1[argwhere(t>=48)].flatten()
v2 = v2[argwhere(t>=48)].flatten()
v3 = v3[argwhere(t>=48)].flatten()

TLS = arange(1., 60, 0.01)

ls1 = LombScargle(t, v1)
power_givenW1    = ls1.power(1./TLS)
ls2 = LombScargle(t, v2)
power_givenW2    = ls2.power(1./TLS)
ls3 = LombScargle(t, v3)
power_givenW3    = ls3.power(1./TLS)

plot(TLS, power_givenW1, color=c_color_palette[0])
plot(TLS, power_givenW2, color=c_color_palette[1])
plot(TLS, power_givenW3, color=c_color_palette[2])
xlabel("Period [h]", fontsize=15)
ylabel("Power Spectral Density", fontsize=15)
vlines(T, 0, max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()) + 0.1*max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()), color="gray", linestyle="--" )
vlines(2.*T, 0, max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()) + 0.1*max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()), color="gray", linestyle=":" )
xticks(arange(0, TLS[-1]+24, 24), fontsize=15)
xlim(0, TLS[-1])
ylim(0, max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()) + 0.1*max(array([power_givenW1, power_givenW2, power_givenW3]).flatten()))
title("I", fontsize=20, loc="left")

tight_layout()

show()
