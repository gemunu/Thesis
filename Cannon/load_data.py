import numpy as np
from TheCannon import dataset
from TheCannon import apogee
from TheCannon import model
from matplotlib import pyplot as plt
from matplotlib import rc
import seaborn as sns
import scipy.stats
import pandas as pd
import itertools
import random

tr=np.load('train_indo.npz',encoding='bytes')
test=np.load('test_indo.npz',encoding='bytes')

wl=np.asarray(tr.items()[0][1][1])

tr_id=[]
for i in tr.items()[0][1][0][:9]:
    tr_id.append(bytes.decode(i))
tr_id=np.asarray(tr_id)

test_id=[]
for i in test.items()[0][1][0]:
    test_id.append(bytes.decode(i))
test_id=np.asarray(test_id)

tr_flux=np.asarray(tr.items()[0][1][2][:9])
test_flux=np.asarray(test.items()[0][1][2])

test_ivar=np.asarray(test.items()[0][1][3])
tr_ivar=np.asarray(tr.items()[0][1][3][:9])

for i in range(len(test_ivar)):
    test_ivar[i]=np.asarray([30000]*len(wl))
for i in range(len(tr_ivar)):
    tr_ivar[i]=np.asarray([30000]*len(wl))


tr_label =apogee.load_labels('trl.csv')

ds = dataset.Dataset(wl, tr_id, tr_flux, tr_ivar, tr_label, test_id, test_flux, test_ivar)
ds.set_label_names(['T_{eff}', '\log g', '[Fe/H]'])
print (ds.tr_ivar,ds.test_ivar)
dls=np.arange(20,110,20)
qs=np.arange(0.1,1,0.1)
fracs=np.arange(0.01,.2,0.02)
fits=np.arange(1,8,1)
fits2=np.arange(1,8,1)
fitm=['sinusoid']
#ds.ranges = [[3840,15011]]#[[1,1338],[1340,3838],[3840,15011]]
ress=[]
combs=list(itertools.product(dls,qs,fracs,fits,fits2,fitm))
random.shuffle(combs)
for comb in combs:
    dl,qv,frv,fv,fv2,fitv= comb

    pseudo_tr_flux, pseudo_tr_ivar = ds.continuum_normalize_training_q(q=qv, delta_lambda=dl)
    contmask = ds.make_contmask(pseudo_tr_flux, pseudo_tr_ivar, frac=frv)

    ds.set_continuum(contmask)
    cont = ds.fit_continuum(fv, fitv)
    norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = ds.continuum_normalize(cont)

    ds.tr_flux = norm_tr_flux
    ds.tr_ivar = norm_tr_ivar
    ds.test_flux = norm_test_flux
    ds.test_ivar = norm_test_ivar

    md = model.CannonModel(fv2)
    md.fit(ds)
    label_errs = md.infer_labels(ds)
    test_labels = ds.test_label_vals
    np.savetxt('indo_test_res.dat',np.c_[test_id,test_labels],fmt='%s')

    trs=pd.read_csv('indo_test_labels')
    trs['ID']=trs['ID'].astype(str)
    test_res=pd.read_csv('indo_test_res.dat',header=None,sep=' ')
    test_res[0]=test_res[0].astype(str).str[:-5]
    res_merge=trs.merge(test_res,left_on='ID',right_on=[0])
    pr=scipy.stats.pearsonr(res_merge['logg'],res_merge[2])[0]
    print (qv,dl,frv,fv,fitv,fv2,pr)
    ress.append((qv,dl,frv,fv,fitv,fv2,pr))


# #
# cfs,waves=[],[]
# for j in range(len(tr_flux)):
#     cf,wlf=[],[]
#     for i in range(len(wl)):
#         if contmask[i]:
#             cf.append(tr_flux[j][i])
#             wlf.append(wl[i])
#     cfs.append(cf)
#     waves.append(wlf)
#
# rc("text", usetex=False)
# for i in range(len(cfs)):
#     plt.plot(waves[i],cfs[i])
#     plt.show()
#
#
#0.4 20 0.01 6 sinusoid 4 0.162546876065
#0.4 100 0.01 7 sinusoid 6 0.115629736714
#0.3 20 0.17 4 sinusoid 3 0.159630194233
#0.4 40 0.19 3 sinusoid 1 0.326985528357
