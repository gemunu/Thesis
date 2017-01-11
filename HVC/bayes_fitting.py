#!/usr/bin/python

#

import os
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as sp
import time
import pandas as pd
import linecache
import scipy.stats

def prob_cal(model_x,obs_x,err_x,model_y,obs_y,err_y,model_m,obs_feh,obs_erf,model_f):
    chi = np.exp(-(((model_x - obs_x) ** 2 / (err_x ** 2)) + ((model_y - obs_y) ** 2/(err_y  ** 2))))
    I = (model_m**(-(2.35))) *chi * scipy.stats.norm(obs_feh, obs_erf).pdf(model_f)
    return (sp.integrate.trapz(I,model_m))

def prob_cal_noimf(model_x,obs_x,err_x,model_y,obs_y,err_y,obs_feh,obs_erf,model_f):
    chi=np.exp(-(((model_x-obs_x)**2/(err_x**2)) + ((model_y-obs_y)**2/(err_y**2))))
    #print (chi)
    #I=(model_m**(-(2.35))) *chi * scipy.stats.norm(obs_feh, obs_erf).pdf(model_f)
    I=chi * scipy.stats.norm(obs_feh, obs_erf).pdf(model_f)
    return I

def load_data(filename,chunksize,iterator=True,csv=False):
    if csv:
        return pd.read_csv(filename,header=None,comment='#',chunksize=None)
    if not iterator:
        return pd.read_csv(filename,header=None,sep=r"\s*",comment='#',chunksize=None)
    else:
        return pd.read_table(filename,header=None,comment='#',iterator=True,chunksize=chunksize,sep=r"\s*")

def log_g(t_star,L_star,):
    return (4 * t_star  - L_star - 10.61)


# def plot_(x,y):
#     n, bins, patches = plt.hist([x, y])
#     plt.show()

# def make_array(i):
#     return np.array
def paras(para_file,model_folder,plot=False):
    obs_data =load_data(para_file,None,iterator=False,csv=False)
    #print (obs_data)
    f1=os.listdir(model_folder)
    #print (f1)
    dats=[]
    for ro,co in obs_data.iterrows():
        #print (co)
        res=[]
        for model_file in f1:
            iso=load_data(model_folder+model_file,chunksize=None,iterator=False)
            #print (model_file,linecache.getline(model_folder+model_file,5).split()[3][6:])
            try:
                model_f=np.float(linecache.getline(model_folder+model_file,5).split()[3][6:])
            except ValueError:
                model_f=np.float(linecache.getline(model_folder+model_file,5).split()[4])
            #print (model_f)
            for col in iso.groupby(np.arange(len(iso))//100):
                #print (col)
                model_tef=col[1][3].mean()#np.array( col[1][3])
                #print (model_tef)
                model_logg=col[1][5].mean()#log_g(col[1][3].mean(),col[1][2].mean())#log_g(np.array(col[1][3]),np.array(col[1][2]))#np.array(col[1][4])
                model_m=np.array(col[1][0])
                #model_m=np.ones(len(np.array(col[1][0])))
                #print (model_tef,np.log10(co[1]))
                #avg_Mv=col[1][4].mean()
                avg_Mv=col[1][4].mean()
                #print (model_file,avg_Mv)
                #prob=prob_cal(model_tef,np.log10(co[1]),100/co[1],model_logg,co[3],0.25,model_m,co[5],0.25,model_f)
                #prob=prob_cal(model_tef,np.log10(co[4]),co[7]/co[4],model_logg,co[5],co[8],model_m,co[6],co[9],model_f)
                #prob=prob_cal(model_tef,np.log10(co[1]),co[2],model_logg,co[3],co[4],model_m,co[5],co[6],model_f)
                prob=prob_cal_noimf(model_tef,np.log10(co[1]),co[2]/co[1],model_logg,co[3],co[4],co[5],co[6],model_f)
                res.append((avg_Mv,prob))

        d1=pd.DataFrame(res)
        d1.columns=['pm','prob']
        #df=d1.groupby('pm').sum()
        #print (d1)
        d1['prob_n']=d1['prob']/d1['prob'].sum()
        #print (co[0],df)
        Mv = (d1['pm'] * d1['prob_n']).sum()/d1['prob_n'].sum()
        errMv=np.sqrt(((d1['prob_n']*( d1['pm'] - Mv)**2).sum())/(d1['prob_n'].sum()))
        dats.append((co[0],Mv,errMv))
        print (co[0],Mv,errMv)
        if plot :
            plt.hist(d1['pm'],weights=d1['prob_n'])
            plt.show()



    #print (dats)
    np.savetxt('paka2',dats,fmt='%s')
    return(dats)


if __name__ == "__main__" :
    #paras('para_g','/Users/gemunu/Research/hvc/iso_sdss/')
    paras('Sp_ace/paras_full_v2.dat','/Users/gemunu/Research/canis/canis/isochrones/')
