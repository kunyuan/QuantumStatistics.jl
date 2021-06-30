import os
import sys
import re
import glob
import math
import traceback
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


a=np.transpose(np.loadtxt("./delta_1.dat",unpack=False))
b=np.transpose(np.loadtxt("./delta_2.dat",unpack=False))
#print(np.amax(np.fabs(b[1]-b_double[1])))
fig, ax = plt.subplots()
plt.figtext(0.01,0.840,"$\Delta_0$",fontsize=8)
plt.figtext(0.92,0.06,r"$Momentum$",fontsize=8)
#plt.figtext(0.70,0.25,r"$\omega_c$=0.25",fontsize=13)

a_max=1.0
b_max=1.0
#a_max=np.amax(np.fabs(a[1]))
#b_max=np.amax(np.fabs(b[1]))
ax.plot(a[0],a[1],'o:',label='new',markersize=2,fillstyle='none')
ax.plot(b[0],b[1],'.-',label='old',markersize=3)
#ax.axvline(x=9.5)
ax.legend(bbox_to_anchor=(0.4,0.5),fontsize=10.0)
# l,b,w,h=0.65,0.26,0.28,0.32
# ax2=fig.add_axes([l,b,w,h])
# ax2.set_yscale('log')
# ax2.plot(a[0,label3:label4],a[1,label3:label4],'o-',markersize=3,fillstyle='none')
#ax2.plot(b[0,label3:label4],b[1,label3:label4],'.-',markersize=3)
#ax.plot(c[0,label2:],c[1,label2:],'s:',label='d-wave',markersize=2)
#plt.ylim([1e-24,1e-1])
plt.savefig('mom_change_wmax.pdf')

c=np.transpose(np.loadtxt("./delta_1_freq.dat",unpack=False))
d=np.transpose(np.loadtxt("./delta_2_freq.dat",unpack=False))

d_max=1.0
#a_max=np.amax(np.fabs(a[1]+a[2]))
#d_max=np.amax(np.fabs(d[1]))
fig, ax = plt.subplots()
plt.figtext(0.01,0.840,"$\Delta_0$",fontsize=8)
plt.figtext(0.92,0.06,r"$Freq$",fontsize=8)
cut_freq=100.0
left_cut = np.searchsorted(c[0], -cut_freq,side='right')
right_cut = np.searchsorted(c[0], cut_freq,side='right')
#plt.figtext(0.70,0.25,r"$\omega_c$=0.25",fontsize=13)
ax.plot(c[0,left_cut: right_cut],c[1,left_cut:right_cut],'o:',label='new',markersize=2,fillstyle='none')
left_cut = np.searchsorted(d[0], -cut_freq,side='right')
right_cut = np.searchsorted(d[0], cut_freq,side='right')
ax.plot(d[0,left_cut: right_cut],d[1,left_cut:right_cut],'.-',label='old',markersize=3)
#ax.axvline(x=9.5)
ax.legend(bbox_to_anchor=(0.4,0.5),fontsize=10.0)
# l,b,w,h=0.65,0.26,0.28,0.32
# ax2=fig.add_axes([l,b,w,h])
# ax2.set_yscale('log')
# ax2.plot(a[0,label3:label4],a[1,label3:label4],'o-',markersize=3,fillstyle='none')
#ax2.plot(b[0,label3:label4],b[1,label3:label4],'.-',markersize=3)
#ax.plot(c[0,label2:],c[1,label2:],'s:',label='d-wave',markersize=2)
#plt.ylim([1e-24,1e-1])
plt.savefig('freq_change_wmax.pdf')



