import pandas as pd
import requests
import urllib
import glob
from bs4 import BeautifulSoup
import sys

names = pd.read_csv('lamost_spectra.dat',header=None)
modNames = [x.strip(' ')+'.gz' for x in list(names[0][:500])]

url = 'http://dr2.lamost.org/sas/fits/'
print(url)


for _ in modNames:
    nam = _.split('_')[0].split('-')[2]
    item = url + nam +'/'
    tf = urllib.request.URLopener()
    print(item+_)
    try:
        tf.retrieve(item + _ , 'lamost_spectra/'+_)
        print(_)

    except urllib.error.HTTPError as err:
       if err.code == 404:
           pass
       else:
               raise

    except KeyboardInterrupt:
        sys.exit()

    print('{0} to go ....'.format(len(modNames) - modNames.index(_)))
