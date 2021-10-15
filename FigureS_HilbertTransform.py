"""
Code underlying Figure S0 of the manuscript
'Principles underlying the complex dynamics of
temperature entrainment by a circadian clock'

Demonstration of how instantaneous phases and
amplitudes are estimated by a Hilbert transformation
"""

from pylab import*
from scipy.signal import hilbert
import numpy as np
import matplotlib.collections as collections
import seaborn as sns


# Set up plotting options
sns.set(style = "ticks")
c_color = [ "tab:blue", "tab:green", "tab:purple" ]
c_ylabel="Conidiation \n Intensity [a.u.]"
c_ylabel_size=15
c_zlabel_size = 20
c_title_size  = 20

# Load time series data
t, v1, v2, v3 = loadtxt("./Data/WT_T22_TP84.txt", unpack=True, usecols=(0,1,2,3))

# Properties of experiment
T = 22
warm = 84               # warm phase in percentage
dummy = 1-(warm/100.)
cold_dur = dummy*T      # cold phase in [h]
t = t/60.               # time array in [h]

# figure specifications
figure(figsize=(10.5/3*2, 2.75*3/3*2))

# subpanel A
subplot(221)

# plot raw time series v1 (replicate 1) and horizontal line at y=0
plot(t, v1, c_color[0], linewidth=2, label="Raw Signal $s(t)$")
hlines(0, 0, t[-1], color="k", linestyle=":")

#
collection = collections.BrokenBarHCollection.span_where(t, ymin=700, ymax=-700, where= ((t % T) <= cold_dur), facecolor="gray", alpha=0.5)
ax = plt.gca()
ax.add_collection(collection)
ylabel("Conidiation Intensity [a.u.]", fontsize=15)
xlabel("Time [h]", fontsize=15)
yticks([-400, -200, 0, 200, 400], ["-400", "", "0", "", "400"])
xticks([24*x for x in range(10)])
xlim(xmin=0, xmax=t[-1])
title("A", fontsize=20, loc="left")

# subpanel B
subplot(222)

# mean detrending
v1=v1-mean(v1)

# Hilbert transformation
hv1 = hilbert(v1)

# plot real and imaginary part of Hilbert transformed signal
plot(hv1.real, hv1.imag, color="k")
hlines(0, -450, 450, color="k", linestyle=":")
vlines(0, -450, 450, color="k", linestyle=":")
xlabel("Raw Signal $s(t)$", fontsize=15)
ylabel("Hilbert Transform of $s(t)$", fontsize=15)
xticks([-400, -200, 0, 200, 400], ["-400", "", "0", "", "400"])
yticks([-400, -200, 0, 200, 400], ["-400", "", "0", "", "400"])
xlim(-450, 450)
ylim(-450, 450)
title("B", fontsize=20, loc="left")

# plot instantaneous amplitude into subpanel A
subplot(221)
plot(t, abs(hv1), linestyle = "--", color="k", label="Amplitude")
legend(loc=0, prop={'size': 9})
ylim(-600, 600)

# subpanel C
subplot(223)

# plot instantaneous conidiation and zeitgeber phase
plot(t, angle(hv1), color="k", label="Conidiation Phase")
plot(t, (2.*pi/T * (t-cold_dur-T/2) )%(2.*pi)-pi, color="gray", label="Zeitgeber Phase")
collection = collections.BrokenBarHCollection.span_where(t, ymin=700, ymax=-700, where= ((t % T) <= cold_dur), facecolor="gray", alpha=0.5)
ax = plt.gca()
ax.add_collection(collection)
hlines(0, 0, t[-1], color="k", linestyle=":")
xlabel("Time [h]", fontsize=15)
xticks([24*x for x in range(10)])
yticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"])
xlim(xmin=0, xmax=t[-1])
ylim(-1.5*pi, 1.5*pi)
xlabel("Time [h]", fontsize=15)
ylabel("Phase [rad]", fontsize=15)
legend(loc="lower left", ncol=1, prop={'size': 9})
title("C", fontsize=20, loc="left")

# subpanel D
subplot(224)

#plot instantaneous conidiation vs zeitgeber phase
scatter((2.*pi/T * (t-cold_dur-T/2) )%(2.*pi)-pi, angle(hv1), alpha=0.5, edgecolors=None, s=5, color=c_color[0], rasterized=True)
hlines(0, -pi, pi, color="k", linestyle=":")
vlines(0, -pi, pi, color="k", linestyle=":")
ylim(-pi, pi)
xlim(-pi, pi)
xticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"])
yticks([-pi, 0, pi], ["$-\pi$", "0", "$\pi$"])
xlabel("Zeitgeber Phase [rad]", fontsize=15)
ylabel("Conidiation Phase [rad]", fontsize=15)
title("D", fontsize=20, loc="left")

tight_layout()

show()
