#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:53:41 2024

@author: marvink
"""

import sys
import os
import glob
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
os.chdir('/home/marvink/Documents/Nansen_Legacy/check_data/field_YSC_pointer')

#%% focus on sic
f = 'SIC.csv'
data = np.genfromtxt(f, delimiter=',', dtype='float64', invalid_raise=False)
# Reshape the data
data = data.reshape((960, 750))
gg=np.array(data)
plt.pcolormesh(gg[0:949,0:739],cmap=plt.cm.coolwarm)
plt.colorbar()
#%%
f = 'SIC_update.csv'
data = np.genfromtxt(f, delimiter=',', dtype='float64', invalid_raise=False)
# Reshape the data
data = data.reshape((960, 750))
gg=np.array(data)
plt.pcolormesh(gg[0:949,0:739],cmap=plt.cm.coolwarm)
plt.colorbar()
#%%
f1 = ['SIC.csv','U.csv','V.csv','T.csv']
i1 = [0,1,2,3]
X =  np.zeros([928,736,4])
X2 = np.zeros([960,750,4])

for i,f in zip(i1,f1):
    gg = pd.read_csv(f,header=None)
    gg = gg.apply(pd.to_numeric,errors='coerce')
    gg = np.array(gg)
    gg = gg.reshape(960,750)
    #gg = np.array(gg)    
    #gg = np.array(gg.transpose()) # due to write_ascii/read_netcdf from fortran, can change, FIXME
    X[:,:,i]  = gg[21:, :736]
    X2[:,:,i] = gg
    if f == 'SIC.csv': # save to not lose Ice over region NN does not cover
        SIC_old = gg
#%%
fig,axs=plt.subplots(1,4, figsize=(28,16))
for i,ax in enumerate(axs):
    ax.pcolormesh(X2[:,:,i])
    
#%%

gg2 = np.zeros([X2.shape[1],X2.shape[0]])
for x in range(X2.shape[0]):
    for y in range(X2.shape[1]):
        gg2[y,x]=X2[x,y,2]
        
        

#%% check mask files

areas = Dataset('areas.nc')
masks = Dataset('masks.nc')
grids = Dataset('grids.nc')
#%%
sall_msk = masks.variables['sall.msk'][:].data
cinn_msk = masks.variables['cinn.msk'][:].data
sall_msk = sall_msk.reshape(960,750)
cinn_msk = cinn_msk.reshape(960,750)

plt.figure()
plt.subplot(121)
plt.pcolormesh(sall_msk)
plt.colorbar()
plt.subplot(122)
plt.pcolormesh(cinn_msk)
plt.colorbar()


#%%


sall_lon = grids.variables['sall.lon'][:].data
sall_lat = grids.variables['sall.lat'][:].data
cinn_lon = grids.variables['cinn.lon'][:].data
cinn_lat = grids.variables['cinn.lat'][:].data
sall_lon = sall_lon.reshape(960,750)
sall_lat = sall_lat.reshape(960,750)
cinn_lon = cinn_lon.reshape(960,750)
cinn_lat = cinn_lat.reshape(960,750)
fig,axs=plt.subplots(1,2, figsize=(16,12),\
                    subplot_kw={'projection':ccrs.NorthPolarStereo(central_longitude=0)})

axs[0].coastlines('10m')
axs[0].pcolormesh(sall_lon,sall_lat,sall_msk,\
                      transform=ccrs.PlateCarree())
axs[1].coastlines('10m')
axs[1].pcolormesh(cinn_lon,cinn_lat,cinn_msk,\
                      transform=ccrs.PlateCarree())

    
    
#%% test writeing out
gg = np.array(data)
gg[np.isnan(gg)] = 0
gg[gg>1] = 0
gg_flat = gg.reshape(960*750)
np.savetxt('SIC_new.csv', gg_flat, delimiter=',', fmt='%.6f')

#%% read in, compare
data1 = np.genfromtxt('SIC_new.csv', delimiter=',', dtype='float64', invalid_raise=False)
# Reshape the data
data1 = data1.reshape((960, 750))
gg1 = np.array(data1)
#%%
plt.pcolormesh(gg-gg1)
