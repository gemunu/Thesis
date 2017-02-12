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
tr=np.load('canis_Train.npz')
test=np.load('canis_Test.npz')

wl=tr['wl']

tr_id=tr['name']
test_id=test['name']

tr_flux=tr['flux']
test_flux=test['flux']

tr_ivar=tr['ivar']
test_ivar=test['ivar']

tr_label =apogee.load_labels('lamost_labels.csv')
ds = dataset.Dataset(wl, tr_id, tr_flux, tr_ivar, tr_label, test_id, test_flux, test_ivar)
ds.set_label_names(['T_{eff}', '\log g', '[Fe/H]'])

ds.ranges = [[3697,5500],[6461,8255]]

pseudo_tr_flux, pseudo_tr_ivar = ds.continuum_normalize_training_q(q=0.90, delta_lambda=50)

contmask = ds.make_contmask(pseudo_tr_flux, pseudo_tr_ivar, frac=0.07)
ds.set_continuum(contmask)
cont = ds.fit_continuum(3, "sinusoid")
norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = ds.continuum_normalize(cont)

ds.tr_flux = norm_tr_flux
ds.tr_ivar = norm_tr_ivar
ds.test_flux = norm_test_flux
ds.test_ivar = norm_test_ivar

md = model.CannonModel(2)
md.fit(ds)

md.diagnostics_leading_coeffs(ds)

md.diagnostics_contpix(ds)

label_errs = md.infer_labels(ds)

test_labels = pd.DataFrame(ds.test_label_vals)

plt.scatter(test_labels[0], test_labels[1])
plt.show()
