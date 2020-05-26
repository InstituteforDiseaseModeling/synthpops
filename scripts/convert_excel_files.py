'''
Converts Excel files to binary for faster loading.
'''

import os
import numpy as np
import pandas as pd
import sciris as sc

folder = os.path.join(sc.thisdir(__file__), os.pardir, 'data', 'demographics', 'contact_matrices_152_countries')
ext1 = '_1.xlsx'
ext2 = '_2.xlsx'
files1 = sc.getfilelist(folder, pattern=f'*{ext1}')
files2 = sc.getfilelist(folder, pattern=f'*{ext2}')

basefilenames = [f[len(folder)+1:-len(ext1)] for f in files1] # 7 is the length of the extension

datadict = {}
for fn in basefilenames:
    datadict[fn] = {}

for fn in basefilenames:
    for i,ext in enumerate([ext1, ext2]):
        thisfile = folder + os.sep + fn + ext
        print(f'Working on {thisfile}...')
        xls = pd.ExcelFile(thisfile)
        for sheet_name in xls.sheet_names:
            if i == 0:
                header = 0
            else:
                header = None
            df = pd.read_excel(xls, sheet_name=sheet_name, header=header)
            arr = np.array(df)
            datadict[fn][sheet_name] = arr

for fn,data in datadict.items():
    thisfile = folder + os.sep + fn + '.obj'
    sc.saveobj(thisfile, data)

print('Done.')