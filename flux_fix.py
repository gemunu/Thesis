
"""
Fix the corrupt fluxes using the gradint of spectrum
"""

import math
import numpy as np      
import os
from astropy.io import fits
from numpy import interp
from matplotlib import pyplot as plt

def plot_(ax,x,y,f,c):
        ax.plot(x, y,c)
        ax.set_title(f)
        plt.draw()

def fluxcor(folder,file_notation,plot_time,save_file=True,plot=True):

	names=os.listdir(folder)
	for star in names:
        	if star.endswith(file_notation):
        	        
		        f=fits.open(folder+star)
	        	print star	        	   
	        	data, header = fits.getdata(folder+star, header=True) 
                	wave = np.arange(data.shape[0]) * header['CDELT1'] + header['CRVAL1']
                	data_grad=np.gradient(data)                	
                	print (wave)
			print (min(data))
			if plot == True:
                                fig, axarr = plt.subplots(2,2,figsize=(20,15))  
	                	plot_(axarr[0,0],wave,data,f=star,c='r')
			        plot_(axarr[0,1],range(1,len(wave)+1),np.gradient(data),star,c='r')
	                        plt.text(0.1, 2.05,'max_grad_index:%s , value %s'%(np.where(np.gradient(data)==max(np.gradient(data)))[0][0],max(np.gradient(data))),size=15,color='b')
                	#print (data_grad)
                	if max(data_grad) > 0.2:
                	        to_drop=np.where(data_grad >.2)[0]
                	        drops=[]
                        	for i in to_drop:
                        	        if i+50 <len(data):
                	                        drops.extend(range(i-50,i+50,1))
        	                        else:
        	                                cango=len(data)-i
        	                                drops.extend(range(i-50,i+cango,1))
                                data[drops]=0.
        	                flux = interp(wave, wave[data>0.], data[data>0.])
	        	        print (min(flux))
				if save_file == True:
				        fits.writeto(f.filename()[:-7]+'.t.fits',flux,header,clobber=True)
                	        if plot == True:
				        plot_(axarr[1,0],wave,flux,star,c='g')
					plt.scatter(wave,flux)

                	elif min(data) < 0. :
                       		flux = interp(wave,wave[data>0],data[data>0])
		        	if save_file == True:
		        		fits.writeto(f.filename()[:-7]+'.t.fits',flux,header,clobber=True)

                        	if plot == True:
					plot_(axarr[1,1],wave,flux,star,c='k')

                        else:
                                try:
					fits.writeto(f.filename()[:-7]+'.t.fits',data,header,clobber=False)
				except IOError:
					pass
                        if plot == True:                                
                		plt.pause(plot_time)
                       		plt.close('all')
                        		
if __name__ == '__main__' :
	#main()
	fluxcor('./data/indous_cool_won/','s.fits',2,save_file=False,plot=True)
