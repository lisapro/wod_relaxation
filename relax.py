'''
Created on 31. jan. 2018

@author: ELP

This is a script for making input data for relaxation for the BROM model 
Raw Source data - Geographically sorted data from World Ocean Database 
https://www.nodc.noaa.gov/OC5/WOD/datageo.html
.nc file created in Ocean Data View program

1. Read the nc file, needed variable. 
https://odv.awi.de/
'''

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

def map_of_stations(ncfile,f_lat,f_lon,title,res):
    fh = Dataset(ncfile, mode='r')
    
    lat_odv = fh.variables['latitude'][:]
    lon_odv = fh.variables['longitude'][:]   
        
    fig  = plt.figure(figsize=(10,6), dpi=100 )
    fh.close()
    gs = gridspec.GridSpec(1,2)
    ax = fig.add_subplot(gs[0]) 
    ax01 = fig.add_subplot(gs[1])
    n = 0.5
    lat_min = (lat_odv.min()-n)
    lat_max = (lat_odv.max()+n)
    lon_min = (lon_odv.min()-n)
    lon_max = (lon_odv.max()+n)
          
    map = Basemap(#width=150000,height=990000,
                resolution= res,projection='merc',\
                lat_0=70,lon_0=14,
                llcrnrlon = lon_min , llcrnrlat = lat_min,
                urcrnrlon = lon_max , urcrnrlat = lat_max,ax = ax)
    
    map2 = Basemap(#width=150000,height=990000,
                resolution='i',projection='stere',\
                lat_0=70,lon_0=14,
                llcrnrlon = 5 ,llcrnrlat = 54 ,
                urcrnrlon = 27 ,urcrnrlat =70,ax = ax01 )
        
    map.drawparallels(np.arange(50.,70.,0.5),labels = [1,1,1,1])
    map.drawmeridians(np.arange(-180.,180.,0.5),labels = [1,1,1,1])
    
    map2.drawparallels(np.arange(50.,70.,5),labels = [1,1,1,1])
    map2.drawmeridians(np.arange(-180.,180.,5),labels = [0,0,1,1])
    
    cont_col = '#bdad95'
    water_col = '#cae2f7'
    map.drawmapboundary(fill_color = water_col)
    map.fillcontinents(color = cont_col)
    map.drawcoastlines(linewidth=0.5)
    
    map2.drawmapboundary(fill_color = water_col)
    map2.fillcontinents(color = cont_col)
    map2.drawcoastlines(linewidth=0.5)
    
    x, y = map(lon_odv, lat_odv)   
    f_lat, f_lon = f_lat, f_lon 
    
    xf1,yf1 = map(f_lon + 0.2,f_lat + 0.1)
    xf2,yf2 = map2(f_lon + 5,f_lat + 1)
    
    map.scatter(x,y,10,marker='o',color='#0b956e',
                alpha = 1, edgecolors = '#054a37')

    map2.scatter(f_lon,f_lat,marker='*', latlon = True,
                  s= 250,color='#eb123a',
                  zorder = 10,edgecolors = "#8d0a22")
        
    map.scatter(f_lon,f_lat,marker='*', latlon = True,
                  s= 250,color='#eb123a',
                  zorder = 10,edgecolors = "#8d0a22") 
           
    #ax.text(xf1,yf1, title, size=15, 
    #        va="center", ha="center", rotation=0,
    #    bbox=dict(boxstyle="Round", alpha=0.7,facecolor = '#cae2f7'))    
         
    ax01.text(xf2,yf2, title, size=15, 
            va="center", ha="center", rotation=0,
        bbox=dict(boxstyle="Round", alpha=0.7,facecolor = '#cae2f7'))

    fig.suptitle('Stations from World Ocean Database')

    if not os.path.exists('data/{}'.format(str(ncfile)[:-3])):
            os.makedirs('data/{}'.format(str(ncfile)[:-3]))
    plt.savefig('data/{}/stations_map.png'.format(str(ncfile)[:-3]))
    #plt.savefig('data/{}/arctic_wod.png'.format()) 
    #plt.show()
    
def choose_month(ds,m_num,var,clima_var_m,levels,double_int = False,int_num = 1):
    # get 1 month data 
    month_ds = ds.where(ds.date_time.dt.month == m_num, drop=True)
    
    # group by depth, find mean
    try:
        month_mean = month_ds.groupby(month_ds['var1']).mean()     
        month_depth = month_mean.var1.values.copy()
        month_df = month_mean.to_dataframe()
        # check for nans 
        nonnan = len(month_df[var]) - np.count_nonzero(np.isnan(month_df[var])) 
        if nonnan > 10 :       
            # interpolate nans 
            var = (month_df[var].interpolate(kind = 'nearest'))
            if np.count_nonzero(np.isnan(var)) > 0: 
                #find first non nan 
                for n,v in enumerate(var):
                    if np.isnan(var[n]) == False: 
                        pos = n                     
                        for i in np.arange(0,pos):
                            var[i] = var[pos]
                        break                           
            #f = interpolate.UnivariateSpline(month_depth,var,k=3)
            f = interpolate.interp1d(month_depth,var)
            var_m = f(levels)                            
            var_m[var_m < 0] = 0
            
            if double_int == True:
                for n in range(0,int_num):
                    f = interpolate.UnivariateSpline(levels,var_m,k=2)
                    var_m = f(levels)
            print ("month with data num: ", m_num)        
        else: 
            #print ('many nans',m_num,month_df[var])
            var_m = clima_var_m
    except:
        var_m = clima_var_m      
    return var_m,month_ds 
       
def create_relax_array(ncfile,varname,pl,save,levels,axis,int_num = 1, double_int = False,only_clima_mean = False):
    
    funcs = {'Oxygen':'var4', 'Temperature': 'var2',
             'si':'var6','alk': 'var12','chl': 'var10',
             'po4':'var5', 'no3':'var7','pH': 'var9'}  
    var_from_odv = funcs[varname] 
    print (varname)      
    # read ncfile as xarray
    ds = xr.open_dataset(ncfile,drop_variables= ('metavar1', 'metavar1_QC',
       'metavar2', 'metavar2_QC', 'metavar3',
       'metavar3_QC', 'longitude', 'longitude_QC', 'latitude', 'latitude_QC',
       'metavar4', 'metavar4_QC', 'metavar5', 'metavar5_QC', 'metavar6',
       'metavar6_QC', 'metavar7', 'metavar7_QC', 'metavar8', 'metavar8_QC',
        'date_time_QC',  'var3_QC', 'var1_QC', 'var2_QC',
       'var4_QC', 'var5_QC',  'var6_QC',  'var7_QC',
       'var8', 'var8_QC', 'var9_QC',  'var10_QC', 'var11',
       'var11_QC',  'var12_QC', 'var13', 'var13_QC', 'var14',
       'var14_QC', 'var15', 'var15_QC', 'var16', 'var16_QC', 'var17',
       'var17_QC', 'var18', 'var18_QC', 'var19', 'var19_QC', 'var20',
       'var20_QC', 'var21', 'var21_QC', 'var22', 'var22_QC', 'var23',
       'var23_QC', 'var24', 'var24_QC', 'var25', 'var25_QC', 'var26',
       'var26_QC', 'var27', 'var27_QC'))
        
    max_depth = np.int(np.max(ds.var1))
    # usually we don't need the data from all depths 
    
    #levels = np.arange(0,350,1)       
    # get only data from 1960 and later 
    if ncfile == 'data_from_WOD_COLLECTION_(2017-08-21T16-04-41).nc':
        ds = ds.where(ds.var5 < 1.2, drop=True)
        ds = ds.where(ds.var6 < 40, drop=True)        
    elif ncfile == 'data_from_WOD_COLLECTION_1500-1600_Smeaheia.nc': 
        ds = ds.where(ds.var4 > 100/44.6, drop=True)    
        #ds = ds.where(ds.var12 > 1, drop=True)    
    # only positive values
    #ds = ds.where(ds[var_from_odv] >= 0, drop=True)
    
    # group by depth and find mean for each depth 
    clima_mean = ds.groupby(ds['var1']).mean()    
    clima_depth = clima_mean.var1.values.copy()
    clima_df = clima_mean.to_dataframe()
    
    # interpolate nans 
    clima_var = (clima_df[var_from_odv].interpolate(kind = 'nearest')) 
    
    # interpolations does not work if 1st variable is nan   
    if np.isnan(clima_var[0]) :
        clima_var[0] = clima_var[1]
        
    # interpolate values to standard levels    
    f = interpolate.interp1d(clima_depth,clima_var) #,k=3
    #            f = interpolate.interp1d(month_depth,var)
    #        var_m = f(levels)
    clima_var_m = f(levels)
    
    if double_int == True:
        for n in range(0,int_num):
            f1 = interpolate.UnivariateSpline(levels,clima_var_m,k=2)
            clima_var_m = f1(levels)
       
    means,depths,days,clima_means = [],[],[],[]
    #depths = [] 
    #days = []
    nday = 0 
    #clima_means = []
    numdays = {1:31,2:28,3:31,4:30,5:31,
               6:30,7:31,8:31,9:30,10:31,
               11:30,12:31} 

    for n in range(1,13):
        print (n)
        var_m, var = choose_month(ds,n,var_from_odv,clima_var_m,levels,double_int,int_num)
        var_m[var_m < 0.] = 0
        #var_m = m[0] 
        #var =  m[1] 
        maxday = numdays[n]
             
        for s in range(0,maxday): 
            nday += 1  
            means.append(var_m)
            depths.append(levels)  
            days.append(np.repeat(nday, len(levels)))
            clima_means.append(clima_var_m)  
    if varname == 'Oxygen':
        means = np.array(means)*44.6     
        clima_var_m = np.array(clima_var_m)*44.6 
        clima_means = np.array(clima_means)*44.6 
        ds[var_from_odv] = ds[var_from_odv]*44.6  
    elif varname == 'alk': 
        means = np.array(means)*1000     
        clima_var_m = np.array(clima_var_m)*1000   
        clima_means = np.array(clima_means)*1000   
        ds[var_from_odv] = ds[var_from_odv]*1000      
    if pl == True:
        
        #plt.clf()
        #axis.set_title(varname)  
        axis.plot(clima_var_m,levels,'ko--',zorder = 10, markersize = 2)
        #ss = np.array(np.array(means).T)
        #print (ss.shape)
        if only_clima_mean  == False: 
            for n in np.arange(0,365,30):
            #    print (n)
                ######axis.scatter(days,depths,c = means)
                axis.plot(np.array(means).T[:,n],np.array(depths).T[:,n],
                 '-',zorder = 8)
            #axis.plot(np.array(means).T[:,0],np.array(depths).T[:,0],
            #     '-',zorder = 8)                
        axis.scatter(ds[var_from_odv],ds['var1'],alpha = 0.5, c = '#7f7f7f' ) #74507c')                
        #plt.savefig('data/{}_relax_{}.png'.format(str(ncfile)[:-3],varname))  
        
        means = np.ravel((np.array(means).T))
        levels_f = np.ravel((np.array(depths).T))
        days = np.ravel((np.array(days).T))
        clima_means = np.ravel((np.array(clima_means).T))  
             
    if only_clima_mean == False:        
        arr = np.vstack((days,levels_f,means)).T
    else: 
        arr = np.vstack((days,levels_f,clima_means)).T
                
    if save == True:    
        if not os.path.exists('data/{}'.format(str(ncfile)[:-3])):
            os.makedirs('data/{}'.format(str(ncfile)[:-3]))

        np.savetxt('data/{}/{}_relax_{}.txt'.format(str(ncfile)[:-3],str(ncfile)[:-3],varname), (arr), delimiter=' ')   
                
    return ds[var_from_odv], ds['var1'], np.array(means).T, np.array(depths).T

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
    df = df[df['var4'] > 3]
    #df = df[df['var7'] < 50]    
    df = df[df.date_time.dt.year > 1940]
    df = df[df.date_time.dt.dayofyear != 129.]
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
             'po4':0.1, 'no3': 0.1,'pH': 0.01}  
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
      
    gridsize = 2   
    xi = np.arange(0,366,gridsize)
    if varname == 'chl':
        yi = np.arange(0,100,gridsize)          
    else:        
        yi = np.arange(0,370,gridsize)  

         
    X,Y = np.meshgrid(xi,yi) 
    
    z_griddata = griddata((day_of_year_nan, depth2_nan),
                   var2_nan2, (xi[None,:], yi[:,None]), method='linear')    
          

    ##f_int2d = interpolate.interp2d(day_of_year_nan,depth2_nan,var2_nan2)
    ###zi_int2d = np.transpose(f_int2d(xi,yi)) 
    f_biv_spline = interpolate.SmoothBivariateSpline(day_of_year_nan, depth2_nan, var2_nan2, kx=kxy, ky=kxy)   
    z_biv_spline = np.transpose(f_biv_spline (xi,yi)) 

    levels = np.arange(vmin,vmax,step)
   
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
   
    CS3 = axis3.scatter(X, Y,c = z_griddata,  alpha = 1,s = 1,cmap = cmap,vmin = vmin, vmax = vmax)
    axis3.set_title('Seasonal variability, griddata')
    
    #CS3 = axis3.contourf(xi, yi,z_griddata,levels = levels ,cmap = cmap,vmin = vmin, vmax = vmax) 
       
    axis4.set_title('Seasonal variability, SmoothBivariateSpline')        
    CS4 = axis4.contourf(xi,yi,z_biv_spline,levels = levels, cmap= cmap) #,vmin = vmin,vmax = vmax) 

    if varname == 'pH': 
        fig.suptitle(r'{}'.format(varname))      
    else:
        fig.suptitle(r'{} $\mu M$'.format(varname))
    
    plt.colorbar(CS3, ax=axis3)
    plt.colorbar(CS4, ax=axis4)    
    for axis in (axis,axis2,axis3,axis4):  
        axis.set_ylim(370,0)
        
    for axis in (axis2,axis3,axis4):       
        axis.set_xlim(0,365)
        axis.set_xlabel('Day in a year')
    fig.savefig('data/{}/smeaheia_wod_2d_{}.png'.format(str(ncfile)[:-3],varname)) 

    plt.clf()
      
def time_to_run():    
    import timeit
    start = timeit.default_timer()
    stop = timeit.default_timer()
    print ('Seconds',stop - start) 

def call_arctic() :
    dss = xr.open_dataset('ROMS_Laptev_Sea.nc')
    levels = sorted(dss.depth.values)
    
    fig  = plt.figure(figsize=(5,8), dpi=100 )
    
    gs = gridspec.GridSpec(3,2)
    gs.update(hspace=0.3,top = 0.95,bottom = 0.05)
    ax = fig.add_subplot(gs[0]) 
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])     
    ax3 = fig.add_subplot(gs[3])     
    ax4 = fig.add_subplot(gs[4])     
    ax5 = fig.add_subplot(gs[5])     
   
    ncfile = 'data_from_WOD_COLLECTION_(2017-08-21T16-04-41).nc'    
    create_relax_array(ncfile,'Oxygen',
                       True, True, levels, ax,1,double_int = True,only_clima_mean = True) # Arctic
    ax.set_title(r'O$_2\ \mu M$')  
    create_relax_array(ncfile,'po4',
                       True, True, levels,ax1,3, double_int = True,only_clima_mean =True) # Arctic,
    ax1.set_title(r'PO$_4\ \mu M$') 
    create_relax_array(ncfile,'si',
                       True, True, levels,ax2,10, double_int = True,only_clima_mean = True) # Arctic
    ax2.set_title(r'Si $\mu M$') 
    create_relax_array(ncfile,'no3',
                       True, True, levels,ax3, 3, double_int = True,only_clima_mean = True) # Arctic
    ax3.set_title(r'NO$_3\ \mu M$') 
    create_relax_array(ncfile,'alk',
                       True, True, levels,ax4, 1, double_int = True,only_clima_mean = True) # Arctic
    ax4.set_title(r'Alkalinity $\mu M$') 
    create_relax_array(ncfile,'pH',
                       True, True, levels,ax5, 3, double_int = True,only_clima_mean = True) # Arctic    
    ax5.set_title(r'pH') 
    #plt.show()
    for axis in (ax,ax1,ax2,ax3,ax4,ax5):
        axis.set_ylim(90,0)
    plt.savefig('data/{}/arctic_wod.png'.format(str(ncfile)[:-3])) 

def call_smeaheia_2d():
    ncfile = 'data_from_WOD_COLLECTION_1500-1600_Smeaheia.nc'
    file = open('data/data_from_WOD_COLLECTION_1500-1600_Smeaheia/smeaheia_Statistics_file.txt','w')    
    #call_smeaheia('2d','po4')
    #call_smeaheia()
    
    #plt.savefig('data/{}/smeaheia_wod.png'.format(str(ncfile)[:-3]))
    
    for_2d_interpolation(ncfile,'Oxygen',file,1)
    for_2d_interpolation(ncfile,'po4',file,1)
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
    
def call_smeaheia() :
    ncfile = 'data_from_WOD_COLLECTION_1500-1600_Smeaheia.nc'
    levels = np.arange(0,300,2)
    save = False
     
    fig  = plt.figure(figsize=(10,6), dpi=100 )    
    gs = gridspec.GridSpec(2,3)
    gs.update(left = 0.04, right = 0.97,hspace=0.3)
    ax = fig.add_subplot(gs[0]) 
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])     
    ax3 = fig.add_subplot(gs[3])     
    ax4 = fig.add_subplot(gs[4])     
    ax5 = fig.add_subplot(gs[5])     
        
    create_relax_array(ncfile,'Oxygen',
                       True, save, levels, ax,1,double_int = True, only_clima_mean = False) 
    ax.set_title(r'O$_2\ \mu M$')   
    #ax.legend(['mean','1','2','3','4','5','6','7','8','9','10','11','12'])
    #ax.legend()
     
    create_relax_array(ncfile,'po4',
                       True, save, levels,ax1,1, double_int = True,only_clima_mean =False)
    ax1.set_title(r'PO$_4\ \mu M$') 
    create_relax_array(ncfile,'si',
                       True, save, levels,ax2,3 , double_int = True,only_clima_mean = False) 
    ax2.set_title(r'Si $ \mu M$') 
    ax2.set_xlim(0,12)
    create_relax_array(ncfile,'no3',
                       True, save, levels,ax3, 6, double_int = True,only_clima_mean = False)
    ax3.set_title(r'NO$_3\ \mu M$') 
    create_relax_array(ncfile,'alk',
                       True, save, levels,ax4, 1, double_int = True,only_clima_mean = False) 
    ax4.set_title(r'Alkalinity $\mu M$') 
    ax4.set_xlim(1900,2600)
    create_relax_array(ncfile,'pH',
                       True, save, levels,ax5, 1, double_int = True,only_clima_mean = False) 
    ax5.set_title(r'pH') 

    for axis in (ax,ax1,ax2,ax3,ax4,ax5):
        axis.set_ylim(450,0)
    #ax.set_ylim(450,0)
    fig.savefig('data/{}/smeaheia_wod.png'.format(str(ncfile)[:-3]),transparent = False)
    #plt.show()           
   
    
def call_jossingfjorden():
    levels = (0,5,10,20,30,50,75,100,125,150,165) #
    fig  = plt.figure(figsize=(10,6), dpi=100 )

    ncfile = 'jossingfjorden-wod.nc'
    gs = gridspec.GridSpec(3,3)
    gs.update(hspace=0.3)
    ax = fig.add_subplot(gs[0]) 
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])     
    ax3 = fig.add_subplot(gs[3])     
    ax4 = fig.add_subplot(gs[4])     
    ax5 = fig.add_subplot(gs[5])
    ax6 = fig.add_subplot(gs[6])    
    
    create_relax_array(ncfile,'Oxygen',
                       True, True, levels, ax,1,double_int = True, only_clima_mean = False) 
    ax.set_title(r'O$_2\ \mu M$')     
    create_relax_array(ncfile,'po4',
                       True, True, levels,ax1,1, double_int = True,only_clima_mean =False)
    ax1.set_title(r'PO$_4\ \mu M$') 
    create_relax_array(ncfile,'si',
                       True, True, levels,ax2,1 , double_int = True,only_clima_mean = False) 
    ax2.set_title(r'Si $ \mu M$') 
    ax2.set_xlim(0,12)
    create_relax_array(ncfile,'no3',
                       True, True, levels,ax3,1, double_int = True,only_clima_mean = False)
    ax3.set_title(r'NO$_3\ \mu M$') 
    create_relax_array(ncfile,'alk',
                       True, True, levels,ax4, 1, double_int = True,only_clima_mean = False) 
    ax4.set_title(r'Alkalinity $\mu M$') 
     
    create_relax_array(ncfile,'pH',
                           True, True, levels,ax5,1, double_int = True,only_clima_mean = False)  
    ax5.set_title(r'pH') 
    
    create_relax_array(ncfile,'chl',
                           True, True, levels,ax6,1, double_int = False ,only_clima_mean = False)     
    ax6.set_title(r'Chlorophyll')
    ax6.set_xlim(0,10)
    for axis in (ax,ax1,ax2,ax3,ax4,ax5,ax6):
        axis.set_ylim(170,0)        
    #plt.show()  
    plt.savefig('data/{}/jossingsfjorden_wod.png'.format(str(ncfile)[:-3])) 
    



def call_osterfjorden():

    ncfile = 'modest-wod.nc'
    
    #map_of_stations(ncfile,60.5,5.3,'Osterfjorden','i')
    levels = np.arange(0,600,2) #(0,5,10,20,30,50,75,100,125,150,165) #
    fig  = plt.figure(figsize=(10,6), dpi=100 )

    gs = gridspec.GridSpec(3,2)
    gs.update(hspace=0.3)
    ax = fig.add_subplot(gs[0]) 
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])     
    ax3 = fig.add_subplot(gs[3])     
    ax4 = fig.add_subplot(gs[4])     
    ax5 = fig.add_subplot(gs[5])
    #ax6 = fig.add_subplot(gs[6])    
    
    create_relax_array(ncfile,'Oxygen',
                       True, True, levels, ax,1,double_int = True, only_clima_mean = False) 
    ax.set_title(r'O$_2\ \mu M$')     
    create_relax_array(ncfile,'po4',
                       True, True, levels,ax1,1, double_int = True,only_clima_mean =False)
    ax1.set_title(r'PO$_4\ \mu M$') 
    create_relax_array(ncfile,'si',
                       True, True, levels,ax2,2 , double_int = True,only_clima_mean = False) 
    ax2.set_title(r'Si $ \mu M$') 
    ax2.set_xlim(0,12)
    
    create_relax_array(ncfile,'no3',
                       True, True, levels,ax3,4, double_int = True,only_clima_mean = False)
    
    ax3.set_title(r'NO$_3\ \mu M$') 
    
    create_relax_array(ncfile,'alk',
                       True, True, levels,ax4, 1, double_int = False,only_clima_mean = False) 
    
    ax4.set_title(r'Alkalinity $\mu M$') 
     
    create_relax_array(ncfile,'pH',
                           True, True, levels,ax5,1, double_int = True,only_clima_mean = False)  
    ax5.set_title(r'pH') 
    
    #create_relax_array(ncfile,'chl',
    #                       True, True, levels,ax6,1, double_int = True ,only_clima_mean = False)     
    #ax6.set_title(r'Chlorophyll')
    #ax6.set_xlim(0,10)
    for axis in (ax,ax1,ax2,ax3,ax4,ax5):
        axis.set_ylim(600,0)        
    #plt.show()  
    plt.savefig('data/{}/Osterfjorden_wod.png'.format(str(ncfile)[:-3]))    


#call_jossingfjorden()    

call_smeaheia_2d()
#call_smeaheia()

#call_arctic()     
#call_osterfjorden()    
#create_relax_array('goldeneye-wod.nc','po4',True, False, levels, double_int = True) #