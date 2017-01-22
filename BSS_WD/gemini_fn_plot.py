import matplotlib.pyplot as plt

temps,fn=[],[]
with open('BSS_final.dat','r') as f:
	for line in f:
		c=line.split()
		temps.append(float(c[5]))
		fn.append(float(c[13]))#- float(c[34]))
lx=[0,4.1]
ly=[8571.4,6228.578]
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=16, usetex=False)
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(fn,temps,'ko',color='0.5',markersize=6)
ax.plot(lx,ly,'r',linestyle='--',linewidth=3)
#ax.plot(4.19,6431.726,'*',markersize=18,color='c',label='UV-normal')
#ax.plot(3.109,6392.488,"*",markersize=18,color='orange',label='UV-excess')
ax.set_xlabel('FUV - NUV (AB mags)')
ax.set_ylabel(r'T$_{eff}\ (K)$')
ax.tick_params(width=2)
ax.set_ylim(7000,9500,500)
#ax.legend(loc='upper right')
#ax.legend(numpoints=1)

#ax.text(4.1, 6600, 'UV-normal', fontsize=15,color='c')

plt.show()
