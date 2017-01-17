#!/usr/bin/python

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


def prob_cal(model_x, obs_x, err_x, model_y, obs_y, err_y, model_m, obs_feh, obs_erf, model_f):
    chi = np.exp(-(((model_x - obs_x)**2 / (err_x**2)) +
                   ((model_y - obs_y)**2 / (err_y**2))))
    I = (model_m**(-(2.35))) * chi * \
        scipy.stats.norm(obs_feh, obs_erf).pdf(model_f)

    return (sp.integrate.trapz(I, model_m))


def prob_cal_noimf(model_x, obs_x, err_x, model_y, obs_y, err_y, obs_feh, obs_erf, model_f):
    chi = np.exp(-(((model_x - obs_x)**2 / (err_x**2)) +
                   ((model_y - obs_y)**2 / (err_y**2))))
    I = chi * scipy.stats.norm(obs_feh, obs_erf).pdf(model_f)
    return I


def load_data(filename, chunksize, iterator=True, csv=False):
    if csv:
        return pd.read_csv(filename, header=None, comment='#', chunksize=None)
    if not iterator:
        return pd.read_csv(filename, header=None, sep=r"\s*", comment='#', chunksize=None)
    else:
        return pd.read_table(filename, header=None, comment='#', iterator=True, chunksize=chunksize, sep=r"\s*")


def log_g(t_star, L_star,):
    return (4 * t_star - L_star - 10.61)


def paras(para_file, model_folder, plot=False):
    obs_data = load_data(para_file, None, iterator=False, csv=True)
    f1 = os.listdir(model_folder)
    #print (f1)
    dats = []
    for ro, co in obs_data.iterrows():
        #print (co)
        res = []
        for model_file in f1:
            iso = load_data(model_folder + model_file,
                            chunksize=None, iterator=False)
            try:
                model_f = np.float(linecache.getline(
                    model_folder + model_file, 5).split()[3][6:])
            except ValueError:
                model_f = np.float(linecache.getline(
                    model_folder + model_file, 5).split()[4])
            #print (model_f)
            for col in iso.groupby(np.arange(len(iso)) // 100):
                model_tef = col[1][3].mean()  # np.array( col[1][3])
                model_logg = log_g(col[1][3].mean(), col[1][2].mean())
                model_m = np.array(col[1][0])
                avg_Mv = col[1][4].mean()
                prob = prob_cal_noimf(model_tef, np.log10(co[1]), co[
                                      2] / co[1], model_logg, co[3], co[4], co[5], co[6], model_f)
                res.append((avg_Mv, prob))

        d1 = pd.DataFrame(res)
        d1.columns = ['pm', 'prob']
        d1['prob_n'] = d1['prob'] / d1['prob'].sum()
        Mv = (d1['pm'] * d1['prob_n']).sum() / d1['prob_n'].sum()
        errMv = np.sqrt(
            ((d1['prob_n'] * (d1['pm'] - Mv)**2).sum()) / (d1['prob_n'].sum()))
        dats.append((co[0], Mv, errMv))
        print(co[0], Mv, errMv)
        if plot:
            plt.hist(d1['pm'], weights=d1['prob_n'])
            plt.show()

    return(dats)


if __name__ == "__main__":
    paras('hvc_vonly_para', '/Users/gemunu/Research/canis/isochrones/')
