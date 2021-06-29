import os
import sys
import re
import glob
import math
import traceback
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


a=np.transpose(np.loadtxt("./test.dat",unpack=False))
b=np.transpose(np.loadtxt("./delta_0_0.dat",unpack=False))
b_double=np.transpose(np.loadtxt("./delta_0_double.dat",unpack=False))
b4=np.transpose(np.loadtxt("./delta_0_4.dat",unpack=False))
b8=np.transpose(np.loadtxt("./delta_0_8.dat",unpack=False))
#print(np.amax(np.fabs(b[1]-b_double[1])))
print(np.amax(np.fabs(b_double[1]-b4[1])))
print(np.amax(np.fabs(b4[1]-b8[1])))
fig, ax = plt.subplots()
plt.figtext(0.01,0.840,"$\Delta_0$",fontsize=8)
plt.figtext(0.92,0.06,r"$Momentum$",fontsize=8)
#plt.figtext(0.70,0.25,r"$\omega_c$=0.25",fontsize=13)
c=np.transpose(np.loadtxt("./delta_1_0.dat",unpack=False))
a_max=1.0
b_max=1.0
#a_max=np.amax(np.fabs(a[1]))
#b_max=np.amax(np.fabs(b[1]))
ax.plot(a[0],a[1]/4/np.pi/np.pi,'o:',label='delta0',markersize=2,fillstyle='none')
ax.plot(a[0],a[2]/4/np.pi/np.pi,'s',label='delta_d',markersize=2,fillstyle='none')
ax.plot(a[0],a[3]/4/np.pi/np.pi,'.',label='delta_tot',markersize=2,fillstyle='none')
print(b[0,np.argmax(np.fabs(c[1])) ] )
ax.plot(b[0],-b[1]/4/np.pi/np.pi,'.-',label='old_0',markersize=3)
ax.plot(c[0],-c[1],'.-',label='old_total',markersize=3)
#ax.axvline(x=9.5)
ax.legend(bbox_to_anchor=(0.4,0.5),fontsize=10.0)
# l,b,w,h=0.65,0.26,0.28,0.32
# ax2=fig.add_axes([l,b,w,h])
# ax2.set_yscale('log')
# ax2.plot(a[0,label3:label4],a[1,label3:label4],'o-',markersize=3,fillstyle='none')
#ax2.plot(b[0,label3:label4],b[1,label3:label4],'.-',markersize=3)
#ax.plot(c[0,label2:],c[1,label2:],'s:',label='d-wave',markersize=2)
#plt.ylim([1e-24,1e-1])
plt.savefig('compare.pdf')


d=np.transpose(np.loadtxt("./delta_1_0.dat",unpack=False))
d[1]=d[1]*4.0*np.pi*np.pi
d_max=1.0
#a_max=np.amax(np.fabs(a[1]+a[2]))
#d_max=np.amax(np.fabs(d[1]))
fig, ax = plt.subplots()
plt.figtext(0.01,0.840,"$\Delta_0$",fontsize=8)
plt.figtext(0.92,0.06,r"$Momentum$",fontsize=8)
#plt.figtext(0.70,0.25,r"$\omega_c$=0.25",fontsize=13)
ax.plot(a[0],a[1]+a[2],'o:',label='new',markersize=2,fillstyle='none')
ax.plot(d[0],-d[1]/d_max*a_max,'.-',label='old',markersize=3)
#ax.axvline(x=9.5)
ax.legend(bbox_to_anchor=(0.4,0.5),fontsize=10.0)
# l,b,w,h=0.65,0.26,0.28,0.32
# ax2=fig.add_axes([l,b,w,h])
# ax2.set_yscale('log')
# ax2.plot(a[0,label3:label4],a[1,label3:label4],'o-',markersize=3,fillstyle='none')
#ax2.plot(b[0,label3:label4],b[1,label3:label4],'.-',markersize=3)
#ax.plot(c[0,label2:],c[1,label2:],'s:',label='d-wave',markersize=2)
#plt.ylim([1e-24,1e-1])
plt.savefig('compare_1.pdf')

e=np.transpose(np.loadtxt("./bare.dat",unpack=False))
fig, ax = plt.subplots()
plt.figtext(0.01,0.840,"$\Delta_0$",fontsize=8)
plt.figtext(0.92,0.06,r"$Momentum$",fontsize=8)
#plt.figtext(0.70,0.25,r"$\omega_c$=0.25",fontsize=13)
ax.plot(e[0],e[1],'.-',label='bare*F',markersize=2,fillstyle='none')
ax.plot(e[0],e[2],'o:',label='bare',markersize=2,fillstyle='none')
ax.plot(e[0],e[3],'s-',label='F',markersize=2,fillstyle='none')
#ax.plot(d[0],d[1],'.-',label='old',markersize=3)
#ax.axvline(x=9.5)
ax.legend(bbox_to_anchor=(0.4,0.5),fontsize=10.0)
# l,b,w,h=0.65,0.26,0.28,0.32
# ax2=fig.add_axes([l,b,w,h])
# ax2.set_yscale('log')
# ax2.plot(a[0,label3:label4],a[1,label3:label4],'o-',markersize=3,fillstyle='none')
#ax2.plot(b[0,label3:label4],b[1,label3:label4],'.-',markersize=3)
#ax.plot(c[0,label2:],c[1,label2:],'s:',label='d-wave',markersize=2)
plt.xlim([0,3.0])
plt.savefig('compare_2.pdf')

