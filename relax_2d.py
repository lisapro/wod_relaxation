import os 
from statsmodels.discrete.tests.test_constrained import junk
from matplotlib import gridspec as gs
from scipy.interpolate import UnivariateSpline  
from scipy import interpolate 
from netCDF4 import Dataset,num2date, date2num
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
import numpy as np       
from datetime import datetime,time  
#import pandas as pd
import xarray as xr
from scipy import interpolate 
import numpy.ma as ma
from scipy.interpolate import griddata



def for_2d_interpolation(ncfile,varname,file,kxy = 1):
    
    fig  = plt.figure(figsize=(10,6), dpi=100 )    
    gs = gridspec.GridSpec(2,2)
    gs.update(left = 0.05,right = 0.97, hspace=0.4, wspace = 0.1)
    axis = fig.add_subplot(gs[0]) 
    axis2 = fig.add_subplot(gs[1]) 
    axis3 = fig.add_subplot(gs[2]) 
    axis4 = fig.add_subplot(gs[3]) 
        
    funcs = {'Oxygen':'var4', 'Temperature': 'var2',
             'si':'var6','alk': 'var12','chl': 'var10',
             'po4':'var5', 'no3':'var7','pH': 'var9'}  
    var_from_odv = funcs[varname] 
    cmap = plt.get_cmap('rainbow')
    
    file.write('\n{}'.format(varname)) 
    
    # read ncfile as xarray
    ds = xr.open_dataset(ncfile)          
    df = ds.to_dataframe()
    #df.var2 = df.var2 * 44.6
    df = df[df['var5'] < 4.5]
    #df = df[df['var2']<(460/44.6)]
    #df = df[df['var7'] < 50]    
    #df = df[df.date_time.dt.year > 1940]
    #df = df[df.date_time.dt.dayofyear != 129.]
    #df = df.date_time.dropna(how='all')
        
    day_of_year = (np.array(df.date_time.dt.dayofyear.values)).flatten()  
    day = (np.array(df.date_time.dt.date.values)).flatten() 
    depth2 = (np.array(df['var1'])).flatten() 
    var2 = (np.array(df[var_from_odv])).flatten()
  
    if varname == 'Oxygen':
        var2 = var2 * 44.6       
    elif varname == 'alk':
        var2 = var2*1000    
    steps = {'Oxygen': 1,'si': 0.1 ,'alk': 1,'chl': 0.01,
             'po4':0.5, 'no3': 0.1,'pH': 0.01}  
    step = steps[varname]     
                          
    #Remove nans 
    depth2_nan = depth2[~np.isnan(depth2)]      
    var2_nan = var2[~np.isnan(depth2)]        
    day_of_year_nan = day_of_year[~np.isnan(depth2)] 

    depth2_nan = depth2_nan[~np.isnan(var2_nan)]      
    var2_nan2 = var2_nan[~np.isnan(var2_nan)]        
    day_of_year_nan = day_of_year_nan[~np.isnan(var2_nan)] 
    
    vmax = np.nanmax(var2_nan2)
    vmin = np.nanmin(var2_nan2)       
    file.write('\nMax over the whole period = {}'.format(vmax))     
    file.write('\nMin over the whole period = {}'.format(vmin))         
      
    gridsize = 1   
    xi = np.arange(0,366,gridsize)
    
    if varname == 'chl':
        yi = np.arange(0,40,gridsize)          
    else:        
        yi = np.arange(0,40,gridsize)  

         
    X,Y = np.meshgrid(xi,yi) 
    
    z_griddata = griddata((day_of_year_nan, depth2_nan),
                   var2_nan2, (xi[None,:], yi[:,None]), method='linear')    
          

    ##f_int2d = interpolate.interp2d(day_of_year_nan,depth2_nan,var2_nan2)
    ###zi_int2d = np.transpose(f_int2d(xi,yi)) 
    f_biv_spline = interpolate.SmoothBivariateSpline(day_of_year_nan, depth2_nan, var2_nan2, kx=kxy, ky=kxy)   
    z_biv_spline = np.transpose(f_biv_spline (xi,yi)) 

    levels = np.arange(vmin,vmax-1,step)
   
    axis.set_title('Long-term variability, raw data')
    for tick in axis.get_xticklabels():
        tick.set_rotation(45)

    CS = axis.scatter(df.date_time.dt.date.values, depth2 ,
                      c = var2,alpha = 1,cmap = cmap,s = 5)   
    plt.colorbar(CS, ax=axis)
    
    axis2.set_title('Seasonal variability, raw data')    
    CS2 = axis2.scatter(day_of_year_nan, depth2_nan, c = var2_nan2,
                         alpha = 1,cmap = cmap,s = 5,vmin = vmin, vmax = vmax)
         
    plt.colorbar(CS2, ax=axis2)     
   
    CS3 = axis3.scatter(X, Y,c = z_griddata,  alpha = 1,s = 2,cmap = cmap) #,vmin = vmin, vmax = vmax)
    axis3.set_title('Seasonal variability, griddata')
    
    #CS3 = axis3.contourf(xi, yi,z_griddata,levels = levels ,cmap = cmap,vmin = vmin, vmax = vmax) 
       
    axis4.set_title('Seasonal variability, SmoothBivariateSpline')        
    #CS4 = axis4.contourf(xi,yi,z_biv_spline, cmap= cmap) #,vmin = vmin,vmax = vmax) 
    CS4 = axis4.pcolormesh(xi,yi,z_griddata, cmap= cmap) #,vmin = vmin,vmax = vmax) 
    
    if varname == 'pH': 
        fig.suptitle(r'{}'.format(varname))      
    else:
        fig.suptitle(r'{} $\mu M$'.format(varname))
    
    plt.colorbar(CS3, ax=axis3)
    plt.colorbar(CS4, ax=axis4)   
    
    
    for axis in (axis,axis2,axis3,axis4):  
        axis.set_ylim(40,0)
        
    for axis in (axis2,axis3,axis4):       
        axis.set_xlim(0,365)
        axis.set_xlabel('Day in a year')
    
    fig.savefig('data/waddensea_wod_2d_{}.png'.format(varname)) 
    #plt.show()
    plt.clf()
    
def call_wadden_2d():
    ncfile = 'wadden_sea.nc'
    #depth = 
    levels = np.arange(0,40,1)
    file = open('data/dwadden_sea.txt','w')    
    #call_smeaheia('2d','po4')
    #call_smeaheia()
    
    #plt.savefig('data/{}/smeaheia_wod.png'.format(str(ncfile)[:-3]))
    for_2d_interpolation(ncfile,'po4',file,1)
    for_2d_interpolation(ncfile,'Oxygen',file,1)
    for_2d_interpolation(ncfile,'pH',file,3)
    for_2d_interpolation(ncfile,'si',file,2)
    for_2d_interpolation(ncfile,'alk',file,1)   
    for_2d_interpolation(ncfile,'chl',file,1) 
    for_2d_interpolation(ncfile,'no3',file,1)
   
    #for_2d_interpolation(ncfile,'po4',ax,ax1,ax2,ax3,file)
    #fig.suptitle(r'PO $_4\ \mu M$')
    #
            
    #axis3.set_title('griddata int gridsize = {}'.format(gridsize))
    
      
    file.close()   

call_wadden_2d()    