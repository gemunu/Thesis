#!/usr/bin/python
import os
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from scipy import trapz
import sys

def get_rsr(folder,band):
	dat=np.loadtxt(folder+band)
	ii=dat[:,0].astype( np.float )	
	jj=dat[:,1].astype( np.float )	
	d=np.array((ii,jj))
	return(d)

def flux_int(data_dir,wl,wh,resw,resr,filename):
	#new_rsr,modelw,modelflux,modelogw = [],[],[],[]
	dats=np.loadtxt(data_dir+filename)
	modelw=[ i for i,j in dats[np.where((dats[:,0]>wl) & (dats[:,0]<wh))]]
	modelflux=[ j for i,j in dats[np.where((dats[:,0]>wl) & (dats[:,0]<wh))]]
	modelogw=np.log10(modelw)
	new_w=sp.interpolate.interp1d(resw,resr)
	#print (min(modelw),min(resw))
	new_rsr= new_w(modelw)
	num = [modelflux[i]*new_rsr[i]*modelw[i]/4  for i in range(len(modelflux))]
	den = [new_rsr[i]*modelw[i] for i in range(len(new_rsr))]
        I1 =sp.trapz(y=num ,x=modelw)
        I2 =sp.trapz(y=den ,x=modelw)
        I= I1/I2
	return I

def mag_(band,flux):
	if band.startswith ('galex1500'):
		mag= (-2.5  * np.log10(flux/1.4e-15)) + 18.82
	elif band.startswith ('galex2500'):
		mag= (-2.5  * np.log10(flux/2.15e-16)) + 20.08
	elif band.startswith('johnson_v'):
		mag= (-2.5 * np.log10(flux)) -21.12
	elif band.startswith('johnson_b'):
		mag= (-2.5 * np.log10(flux)) -20.45
	elif band.startswith('johnson_u'):
		mag= (-2.5 * np.log10(flux)) -20.94		
	elif band.startswith('u.'):
		mag= (-2.5  * np.log10(flux)) -21.1+.037
	elif band.startswith('g.'):
		mag= (-2.5  * np.log10(flux)) -21.1-.01
	elif band.startswith('r.'):
		mag= (-2.5  * np.log10(flux)) -21.1+.003
	elif band.startswith('i.'):
		mag= (-2.5  * np.log10(flux)) -21.1-.006
	elif band.startswith('z.'):
		mag= (-2.5  * np.log10(flux)) -21.1-.016
	elif band.startswith('2massj'):
		mag= (-2.5  * np.log10(flux)) -.001 + 2.5*np.log10(3.129e-10) - 0.017
	elif band.startswith('2massh.'):
		mag= (-2.5  * np.log10(flux)) +.019 + 2.5*np.log10(1.133e-10) + 0.016
	elif band.startswith('2massk.'):
		mag= (-2.5  * np.log10(flux)) -.017 + 2.5*np.log10(4.283e-11) + 0.003
	else:
		print('This is just stmag')
		mag= (-2.5 * np.log10(flux)) -21.1
	return mag
	        
def flux(data_dir,rsr_dir,band,m_or_f=True):
	#print (data_dir)
	try:
		wave=get_rsr(rsr_dir,band)[0]
	except:
		print ("No such band !")
		sys.exit(0)
	resp=get_rsr(rsr_dir,band)[1]
	f1=os.listdir(data_dir)
	wl=min(wave)
	wh=max(wave)
	#print (wl,wh)
	magar=[]
	for eachfile in f1:
		#print (eachfile)
		band_flux=flux_int(data_dir,wl,wh,wave,resp,eachfile)
		if m_or_f :
			magnitude=mag_(band,band_flux)	
			print eachfile,band,band_flux,magnitude	
			magar.append(eachfile)#,magnitude)
			#magar.append(magnitude)
		if not m_or_f:
			print (eachfile,band,band_flux)	
			magar.append(band_flux)
	return (magar)
	
if __name__ == '__main__' :
	flux("./kurucz04_fluxes/","./rsrs/","galex2500_2.dat",True)

