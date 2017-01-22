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
import matplotlib.image as mpimg

rc("text", usetex=False)
tr=np.load('train_indo.npz',encoding='bytes')
test=np.load('test_indo.npz',encoding='bytes')

wl=np.asarray(tr.items()[0][1][1][3838:8088])

tr_id=[]
for i in tr.items()[0][1][0]:
    tr_id.append(bytes.decode(i))
tr_id=np.asarray(tr_id)

test_id=[]
for i in test.items()[0][1][0]:
    test_id.append(bytes.decode(i))
test_id=np.asarray(test_id)

tr_flux=np.asarray(tr.items()[0][1][2])
test_flux=np.asarray(test.items()[0][1][2])

test_ivar=np.asarray(test.items()[0][1][3])
tr_ivar=np.asarray(tr.items()[0][1][3])

test_ivar=test_ivar[0:1509,3838:8088]
tr_ivar=tr_ivar[0:1509,3838:8088]

tr_flux=tr_flux[0:1509,3838:8088]
test_flux=test_flux[0:1509,3838:8088]

for i in range(len(test_ivar)):
    test_ivar[i]=np.asarray([23000]*len(wl))#np.random.uniform(10000,40000,len(wl))#np.asarray([23000]*len(wl))
for i in range(len(tr_ivar)):
    tr_ivar[i]=np.asarray([23000]*len(wl))#np.random.uniform(10000,40000,len(wl))#np.asarray([23000]*len(wl))


tr_label =apogee.load_labels('train_labels.csv')#[:30]
#tr_label =apogee.load_labels('trl.csv')
test_id=tr_id
test_flux=tr_flux
test_ivar=tr_ivar
ds = dataset.Dataset(wl, tr_id, tr_flux, tr_ivar, tr_label, test_id, test_flux, test_ivar)
ds.set_label_names(['T_{eff}', '\log g', '[Fe/H]'])
print (ds.tr_ivar,ds.test_ivar)
dls=np.arange(10,110,20)
qs=np.arange(0.1,1,0.1)
fracs=np.arange(0.01,.2,0.02)
fits=np.arange(1,8,1)
fits2=np.arange(1,8,1)
fitm=['sinusoid']
#ds.ranges = [[100,3500],[3838,8088],[11338,13838]]
#ress=[]
#combs=list(itertools.product(dls,qs,fracs,fits,fits2,fitm))
# random.shuffle(combs)
# for comb in combs:
qv,dl,frv,fv,fitv,fv2=(0.9, 50, 0.1, 3 ,'chebyshev', 2)
def image_(img):
    image = mpimg.imread(img)
    plt.imshow(image)
    plt.show()

#combs=list(itertools.product(dls,qs,fracs,fits,fits2,fitm))
#(qv,dl,frv,fv,fitv,fv2,pr)
combs=[(10, 0.7, 0.03,5,2,'sinusoid')]
random.shuffle(combs)
for comb in combs:
    dl,qv,frv,fv,fv2,fitv = comb
    pseudo_tr_flux, pseudo_tr_ivar = ds.continuum_normalize_training_q(q=qv, delta_lambda=dl)
    contmask = ds.make_contmask(pseudo_tr_flux, pseudo_tr_ivar, frac=frv)

    ds.set_continuum(contmask)
    cont = ds.fit_continuum(fv, fitv)
    norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = ds.continuum_normalize(cont)

    ds.tr_flux = norm_tr_flux
    ds.tr_ivar = norm_tr_ivar
    ds.test_flux = norm_test_flux
    ds.test_ivar = norm_test_ivar

    for i in norm_tr_flux:
        i[(wl>5400) & (wl<5500)]=0
    for i in norm_test_flux:
        i[(wl>5400) & (wl<5500)]=0

    md = model.CannonModel(fv2)
    md.fit(ds)
    label_errs = md.infer_labels(ds)
    test_labels = ds.test_label_vals
    # ds.diagnostics_1to1()
    # image_('1to1_label_0.png')
    # image_('1to1_label_1.png')
    # image_('1to1_label_2.png')
    np.savetxt('indo_test_res_2.dat',np.c_[test_id,test_labels],fmt='%s')

    trs=pd.read_csv('train_labels.csv')
    trs['ID']=trs['ID'].astype(str)
    test_res=pd.read_csv('indo_test_res_2.dat',header=None,sep=' ')
    test_res[0]=test_res[0].astype(str).str[:-5]
    res_merge=trs.merge(test_res,left_on='ID',right_on=[0])
    pr=scipy.stats.pearsonr(res_merge['logg_{corr}'],res_merge[2])[0]
    print (qv,dl,frv,fv,fitv,fv2,pr)
    #ress.append((qv,dl,frv,fv,fitv,fv2,pr))
    # sns.distplot(res_merge['logg_{corr}']-res_merge[2],kde=None)
    # plt.show()

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

#0.7 10 0.03 5 sinusoid 2 0.281274812801
