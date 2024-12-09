#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:21:23 2024

@author: marvink
"""
import sys
import os
import glob
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path

print('HERE 0')
# for error logging
#try:
sys.path.append(os.getcwd())
from unet import create_ResidualUNET
from Dataset.Dataloader import TestBarentsGenerator
from CommonFunctions import read_config_from_csv, load_test_data

HM_LIB='/perm/famk/hm_lib/CY43_cpl_NN/lib/util/NN/' #FIXME
path_dataset=HM_LIB+'/Dataset/Data/'
path_predictions=HM_LIB+'/Predictions/'

#%% create config
config = read_config_from_csv(HM_LIB+'26032247.csv')

#%% create model
model = create_ResidualUNET(
        input_shape = (config['height'], config['width'], len(config['fields'])),
        channels = config['channels'],
        pooling_factor = config['pooling_factor']
    )

#%% load weights
weights = '26032247'
wpath=HM_LIB+'/outputs/models/'+weights
load_status = model.load_weights(wpath).expect_partial()

# mins and maxs for weighting
mins = [0.0,
        -1.8776785135269165,
        -1.7206077575683594,
        -31.46680914938448,
        -36.95461884462837,
        230.54005432128906,
        0.0]
maxs = [1.0,
        1.7968040704727173,
        2.0257761478424072,
        34.56073335529755,
        32.089516519829104,
        304.1977844238281,
        1.0]

#%% create X for running NN
# order of fields is important in X!!
# the field names / file names for reading in
#f1 = ['SFX_SIC','SFX_X10M','SFX_Y10M','SFX_T2M']
f1 = ['SIC.csv','U.csv','V.csv','T.csv']
i1 = [0,3,4,5]
f2 = ['ice_u_input', 'ice_v_input','lsmask']
i2 = [1,2,6]
X = np.zeros([1,928,736,7])
# need to take ice drift from Sample file, atm fields from AA file
drift = Dataset(HM_LIB+'/Dataset/Data/2023/01/01/T00Z/CouplingSample_v20230101T12_b20230101T00.nc')
#AAmod = Dataset(os.getcwd()+'/AAfields.nc')
# write on X
for i,f in zip(i2,f2):
    gg = drift.variables[f][:,:].data
    X[0,:,:,i] = gg[21:,:736]

for i,f in zip(i1,f1):
    data = np.genfromtxt(HM_LIB+f, delimiter=',', dtype='float64', invalid_raise=False)
    # Reshape the data
    data = data.reshape((960, 750))
    gg=np.array(data)
    if f=='SIC.csv':
        gg[np.isnan(gg)] = 0 # replace nan mask over land with 0
        gg[gg>1] = 0         # make sure ice field has no values exceeding 1
        SIC_old  = gg        # save sea ice field
    # write on array given to NN
    X[0,:,:,i] = (gg[32:, :736] - mins[i]) / (maxs[i]-mins[i])


#%% make NN prediction 
y_pred = np.clip(model.predict(X), 0, 1)
print('NN run finished!')
#%% save as ascii
# Initialize the full field array with zeros
full_field = np.zeros(SIC_old.shape)

# Copy y_pred into the correct position of the full field
full_field[32:32+y_pred.shape[1], :736] = y_pred[0,:,:,0]
#SIC_old = drift.variables['sic_input'][:,:].data
# update full_field where SIC_old has values > 0 to not loose ice where NN does not cover
for i in range(32):
    for j in range(SIC_old.shape[1]):
        if SIC_old[i, j] > 0:
            full_field[i, j] = SIC_old[i, j]
# write to file
#df = pd.DataFrame(full_field)
#df.to_csv('SIC_update.csv', index=False,header=False)
gg_flat = full_field.reshape(960*750)
np.savetxt(HM_LIB+'SIC_update.csv', gg_flat, delimiter=',', fmt='%.6f')

print('Python: SIC_updated[900,700] is: ', full_field[899,699])


#except Exception as e:
#    # Write the error message to a file if an exception occurs
#    with open(os.getcwd()+'\python_error_log.txt', 'w') as f:
#        f.write(f"An error occurred: {str(e)}")


