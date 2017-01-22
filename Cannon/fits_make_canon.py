import numpy as np
import os
from astropy.io import fits
from numpy import interp

fold=list(set(os.listdir('./indou/')))
nam=np.genfromtxt('indous_35l.dat',dtype=str)
results=[]

kk=[]
for i in nam:
	for j in fold:
		if i in j:
			kk.append(j)
wave=np.arange(3465.,9469.1,0.4)
fs,ers,names=np.empty(0),np.empty(0),np.empty(0,dtype=str)

for k in kk:

	dats=fits.open('./indou/'+k)
	names=np.append(names,k)
	#wave=dats[1].data[0]
	flux=dats[1].data[0][1]
	ww=np.empty(0)
	ff=np.empty(0)
	for i in range(2,len(dats),1):
		ww=np.append(ww,dats[i].data[0][0])
		ff=np.append(ff,dats[i].data[0][4])
	er_flux=interp(wave,ww,ff)
	er_ivar=1/(er_flux**2)
	fs=np.append(fs,flux)
	ers=np.append(ers,er_ivar)

results.append(names)
results.append(wave)
results.append(fs)
results.append(ers)
np.savez('test',results)

# tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='star', format='20A', array=names),fits.Column(name='wave', format='E', array=wave)\
# ,fits.Column(name='flux', format='E', array=fs),fits.Column(name='er_flux', format='E', array=ers)])
# print (tbhdu)
# tbhdu.writeto('test.fits')
